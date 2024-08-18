"""
TODO: fix training mixed precision -- issue with AdamW optimizer
"""

import argparse
import logging
import math
import os
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision.transforms import v2  # from vision transforms
from tqdm.auto import tqdm
from pytorch_optimizer import CAME
import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available

from safetensors.torch import save_file

import lpips
from PIL import Image

import config # this is a local file config.py
from arg_parser import parse_basic_args

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)

def convert_vae_state_dict(vae_state_dict):
    vae_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3-i}.upsample."
            vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

    # this part accounts for mid blocks in both the encoder and the decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i+1}."
        vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

    if diffusers.__version__ < "0.17.0":
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "query."),
            ("k.", "key."),
            ("v.", "value."),
            ("proj_out.", "proj_attn."),
        ]
    else:
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "to_q."),
            ("k.", "to_k."),
            ("v.", "to_v."),
            ("proj_out.", "to_out.0."),
        ]

    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                # logger.info(f"Reshaping {k} for SD format: shape {v.shape} -> {v.shape} x 1 x 1")
                new_state_dict[k] = reshape_weight_for_sd(v)

    return new_state_dict

# accelerator's unwrap_model is not working (might be a version issue with diffuser bc I'm using the latest ver),
# so I asked chat gpt to give me the function:

def acc_unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (e.g., DDP, FSDP, etc.).

    Args:
        model (torch.nn.Module): The model to unwrap.

    Returns:
        torch.nn.Module: The unwrapped model.
    """
    # If the model is wrapped by torch's DDP (DistributedDataParallel), return the original model.
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    # If the model is wrapped by torch's FSDP (FullyShardedDataParallel), return the original model.
    elif hasattr(model, "_fsdp_wrapped_module"):
        return unwrap_model(model._fsdp_wrapped_module)
    # If the model is wrapped by torch's AMP (Automatic Mixed Precision) or other wrappers, return the original model.
    else:
        return model

if is_wandb_available():
    import wandb
    wandb.login(key=config.wandb_api_key)

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(args, test_dataloader, vae, accelerator, weight_dtype, epoch=0, repo_id=None, curr_step = 0):
    logger.info("Running validation... ")

    vae_model = acc_unwrap_model(vae)
    images = []
    
    for _, sample in enumerate(test_dataloader):
        x = sample["pixel_values"].to(weight_dtype)
        reconstructions = vae_model(x).sample
        images.append(
            torch.cat([sample["pixel_values"].cpu(), reconstructions.cpu()], axis=0)
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "Original (left), Reconstruction (right)", np_images, epoch
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left), Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                },
                step=curr_step
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.gen_images}")

    if args.push_to_hub and repo_id:
        try:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
        except:
            logger.info(f"UserWarning: Your huggingface's memory is limited. The weights will be saved only local path : {args.output_dir}")
    
    del vae_model
    torch.cuda.empty_cache()

def make_image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_model_card(args, repo_id: str, images=None, repo_folder=None):
    # img_str = ""
    # if len(images) > 0:
    #     image_grid = make_image_grid(images, 1, "example")
    #     image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
    #     img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: Nothing: \n

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def parse_args():
    # I moved the argument parser to a separate file, although active, I edit the params below
    parser = argparse.ArgumentParser(description="Simple example of a VAE training script.")
    parser = parse_basic_args(parser)
    args = parser.parse_args()

    # default setting I'm using:
    args.pretrained_model_name_or_path = r"pretrained_model/sdxl_vae"
    # args.revision
    #args.dataset_name = None
    #args.dataset_config_name
    args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/sample"
    #args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/train"
    args.test_data_dir = r"/home/wasabi/Documents/Projects/data/vae/test"
    #args.image_column
    args.output_dir = r"outputs/models"
    #args.huggingface_repo
    #cache_dir =
    args.seed = 420
    args.resolution = 1024
    args.train_batch_size = 2 # batch 2 was the best for a single rtx3090
    args.num_train_epochs = 1
    args.gradient_accumulation_steps = 3
    args.gradient_checkpointing = True
    args.learning_rate = 1e-07
    args.scale_lr = True
    args.lr_scheduler = "constant"
    args.max_data_loader_n_workers = 2
    args.lr_warmup_steps = 0
    args.logging_dir = r"outputs/vae_log"
    args.mixed_precision = 'bf16'
    args.report_to = None #'wandb'
    args.checkpointing_steps = 50# return to 5000
    #args.checkpoints_total_limit
    #args.resume_from_checkpoint
    args.test_samples = 20
    args.validation_epochs = 1
    args.tracker_project_name = "vae-fine-tune"
    args.use_8bit_adam = False
    args.use_ema = True # this will drastically slow your training, check speed vs performance
    #args.kl_scale
    args.push_to_hub = False
    #hub_token

    #following are new parameters
    args.use_came = False
    args.diffusers_xformers = True
    args.save_for_SD = True
    args.save_precision = "fp16"

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

# def preprocess(examples):
#     images = [image.convert("RGB") for image in examples["image"]]
#     examples["pixel_values"] = [train_transforms(image) for image in images]
#     return examples

# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#     return {"pixel_values": pixel_values}

def get_all_images_in_folder(folder, image_ext=(".png", ".jpeg", ".jpg", ".PNG", ".JPEG", ".JPG"), rejected_folders=[]):
    """
    return all images paths from a certain folder, recursively
    Args:
        rejected_folders: list of rejected names that could be inside folders
        folder:
        image_ext:

    Returns: list[str]
    """
    S = []
    for f in os.listdir(folder):
        if not any([x in f for x in rejected_folders]):
            if f.endswith(image_ext):
                S.append(os.path.join(folder, f))
            elif os.path.isdir(os.path.join(folder, f)):
                S.extend(get_all_images_in_folder(os.path.join(folder, f), image_ext, rejected_folders))
    return S


def main():
    # clear any chache
    torch.cuda.empty_cache()

    args = parse_args()

    # if not os.path.exists(os.path.join(args.dataset_name, "train\\metadata.jsonl")):
    #    fnames = get_all_images_in_folder(args.dataset_name)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)



    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        repo_id = None 
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(args.huggingface_repo).name, exist_ok=True, token=args.hub_token
            ).repo_id


    def get_dtype(str_type=None):
        # return torch dtype
        torch_type = torch.float32
        if str_type == "fp16":
            torch_type = torch.float16
        elif str_type == "bf16":
            torch_type = torch.bfloat16
        return torch_type

    weight_dtype = get_dtype(accelerator.mixed_precision)
    
    save_for_SD = False
    if args.save_for_SD:
        save_for_SD = True
        save_dtype = get_dtype(args.save_precision)
        
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float32
        )
    except:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32
        )

    vae.requires_grad_(True)

    # https://stackoverflow.com/questions/75802877/issues-when-using-huggingface-accelerate-with-fp16
    # load params with fp32, which is auto casted later to mixed precision, may be needed for ema
    #
    # from the overflow's answer, it links to the diffuser's sdxl training script example and in the code there's another link
    # which points to https://github.com/huggingface/diffusers/pull/6514#discussion_r1447020705
    # which may suggest we need to do all this casting before passing the learnable params to the optimizer
    for param in vae.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    # Load vae
    if args.use_ema:
        try:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float32)
        except:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)

        # no need to do special loading
        #for param in ema_vae.parameters():
        #    if param.requires_grad:
        #        param.data = param.to(torch.float32)
    # imported xformers from kohya's sdxl train, reading the JP, it seems like vae training can benefit from xformers:

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
        #if args.use_ema:
        #    set_diffusers_xformers_flag(ema_vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        print("loading with better accelerate")
         
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(vae, weights, output_dir):
            if args.use_ema:
                ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

            logger.info(f"{vae = }")
            
            # it seems like the vae/model at this point is a list, thus we index it at 0
            vae = vae[0]
            vae.save_pretrained(os.path.join(output_dir, "vae"))

            # that saves it for diffuser

            if save_for_SD:
                # now we need an additional method for saving for Stable diffusion
                state_dict = {}
                def update_sd(prefix, sd):
                    for k, v in sd.items():
                        key = prefix + k
                        if save_dtype is not None:
                            v = v.detach().clone().to("cpu").to(save_dtype)
                        state_dict[key] = v
                
                #model_dict = vae.state_dict()
                vae_dict = convert_vae_state_dict(vae.state_dict())
                update_sd("", vae_dict)
                save_file(state_dict, os.path.join(output_dir, "vae_sdxl_ft.safetensors"))


        def load_model_hook(vae, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), AutoencoderKL)
                ema_vae.load_state_dict(load_model.state_dict())
                ema_vae.to(accelerator.device, dtype=weight_dtype)
                del load_model

            # load diffusers style into model
            load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae", weight_dtype=torch.float32)
            vae.register_to_config(**load_model.config)

            vae.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        # Prepare everything with our `accelerator`.

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.encoder.to(accelerator.device, dtype=weight_dtype)
    vae.decoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes for 8-bit Adam. Run `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows")

        optimizer_cls = bnb.optim.AdamW8bit
        optimizer = optimizer_cls(
            vae.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    elif args.use_came:
        optimizer_cls = CAME
        optimizer = optimizer_cls(
            vae.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999),
            weight_decay=0.01
        )
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            vae.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    train_transforms = v2.Compose(
        [
            v2.Resize(
                args.resolution, interpolation=v2.InterpolationMode.BILINEAR
            ),
            v2.RandomCrop(args.resolution),

            #v2.ToTensor(), # this is apparently going to be depreciated in the future, replacing with the following 2 lines
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            
            v2.Normalize([0.5], [0.5]),
            #v2.ToDtype(weight_dtype)
        ]
    )

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    # def test_preprocess(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     return examples

    with accelerator.main_process_first():
        # Load test data from test_data_dir
        if (args.test_data_dir is not None and args.train_data_dir is not None):
            logger.info(f"load test data from {args.test_data_dir}")
            test_dir = os.path.join(args.test_data_dir, "**")
            test_dataset = load_dataset(
                "imagefolder",
                data_files=test_dir,
                cache_dir=args.cache_dir,
            )
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = test_dataset["train"].with_transform(preprocess)
        # Load train/test data from train_data_dir
        elif "test" in dataset.keys():
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)
        # Split into train/test
        else:
            dataset = dataset["train"].train_test_split(test_size=args.test_samples)
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format, dtype=weight_dtype)  # .float()
        return {"pixel_values": pixel_values}

    # persistent_workers=args.persistent_data_loader_workers,
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=n_workers  # args.train_batch_size*accelerator.num_processes,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=1,  # args.train_batch_size*accelerator.num_processes,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    (
        vae, vae.encoder, vae.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    ) = accelerator.prepare(
        vae, vae.encoder, vae.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_vae.to(accelerator.device, dtype=weight_dtype)
        ema_vae = accelerator.prepare(ema_vae)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"args.max_train_steps {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            # dirs = os.listdir(args.output_dir)
            dirs = os.listdir(args.resume_from_checkpoint)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(os.path.join(path))  # kiml
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device, dtype=weight_dtype)
    lpips_loss_fn.requires_grad_(False)
    lpips_loss_fn.eval()  # added
    print("num porcess", accelerator.num_processes)
    print("working with", weight_dtype)
    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        accelerator.wait_for_everyone()
        train_loss = 0.0
        logger.info(f"{epoch = }")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(vae):
                with accelerator.autocast():
                    target = batch["pixel_values"]  # .to(accelerator.device, dtype=weight_dtype)
                    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py

                    if accelerator.num_processes > 1:
                        posterior = vae.module.encode(target).latent_dist
                        z = posterior.sample()
                        pred = vae.module.decode(z).sample
                    else:
                        posterior = vae.encode(target).latent_dist  # .to(weight_dtype)
                        # z = mean                      if posterior.mode()
                        # z = mean + variable*epsilon   if posterior.sample()
                        z = posterior.sample()  # .to(weight_dtype) # Not mode()
                        pred = vae.decode(z).sample

                    # pred = pred#.to(dtype=weight_dtype)
                    kl_loss = posterior.kl().mean()  # .to(weight_dtype)

                    # mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                    mse_loss = F.mse_loss(pred, target, reduction="mean")
                    with torch.no_grad():
                        lpips_loss = lpips_loss_fn(pred, target).mean()
                        if not torch.isfinite(lpips_loss):
                            lpips_loss = torch.tensor(0)

                    loss = (
                            mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
                    )  # .to(weight_dtype)

                    if not torch.isfinite(loss):
                        pred_mean = pred.mean()
                        target_mean = target.mean()
                        logger.info("\nWARNING: non-finite loss, ending training ")

                        accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_vae.step(vae.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": mse_loss.detach().item(),
                "lpips": lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                with torch.no_grad():
                    log_validation(args, test_dataloader, vae, accelerator, weight_dtype, epoch, repo_id, global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
    
