"""
TODO: fix training mixed precision -- issue with AdamW optimizer
"""

import argparse
import logging
import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

from convert_to_SD import convert_vae_state_dict

import config # this is a local file config.py
from arg_parser import parse_basic_args

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)


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

# I asked chatgpt-4o to give me ideas for better vae (with better reconstruction of smaller details, aka hands and faces)


# Function to split the image into patches
def extract_patches(image, patch_size, stride):
    # Unfold the image into patches
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # Reshape to get a batch of patches
    patches = patches.contiguous().view(image.size(0), image.size(1), -1, patch_size, patch_size)
    return patches

# Patch-Based MSE Loss
def patch_based_mse_loss(real_images, recon_images, patch_size=32, stride=16):
    real_patches = extract_patches(real_images, patch_size, stride)
    recon_patches = extract_patches(recon_images, patch_size, stride)
    mse_loss = F.mse_loss(real_patches, recon_patches)
    return mse_loss

# Patch-Based LPIPS Loss (using the pre-defined LPIPS model)
def patch_based_lpips_loss(lpips_model, real_images, recon_images, patch_size=32, stride=16):
    with torch.no_grad():
        real_patches = extract_patches(real_images, patch_size, stride)
        recon_patches = extract_patches(recon_images, patch_size, stride)
        
        lpips_loss = 0
        # Iterate over each patch and accumulate LPIPS loss
        for i in range(real_patches.size(2)):  # Loop over number of patches
            real_patch = real_patches[:, :, i, :, :].contiguous()
            recon_patch = recon_patches[:, :, i, :, :].contiguous()
            patch_lpips_loss = lpips_model(real_patch, recon_patch).mean()
            
            # Handle non-finite values
            if not torch.isfinite(patch_lpips_loss):
                patch_lpips_loss = torch.tensor(0, device=real_patch.device)
            
            lpips_loss += patch_lpips_loss

    return lpips_loss / real_patches.size(2)  # Normalize by the number of patches


if is_wandb_available():
    import wandb
    wandb.login(key=config.wandb_api_key)

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(args, test_dataloader, vae, accelerator, weight_dtype, epoch=0, repo_id=None, curr_step = 0, max_validation_sample=20):
    logger.info("Running validation... ")

    vae_model = acc_unwrap_model(vae)
    images = []
    
    for i, sample in enumerate(test_dataloader):
        if i < max_validation_sample:
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
    
    #args.image_column
    args.output_dir = r"outputs/models_v3"
    #args.huggingface_repo
    #cache_dir =
    args.seed = 420
    args.resolution = 1024
    args.train_batch_size = 2 # batch 2 was the best for a single rtx3090
    args.num_train_epochs = 5
    args.gradient_accumulation_steps = 3
    args.gradient_checkpointing = True
    args.learning_rate = 1e-04
    args.scale_lr = True
    args.lr_scheduler = "constant"
    args.max_data_loader_n_workers = 4
    args.lr_warmup_steps = 0
    args.logging_dir = r"outputs/vae_log"
    args.mixed_precision = 'fp16'
    
    
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
    args.lpips_scale = 5e-1
    args.kl_scale = 1e-6 # this is not relevant with patched loss
    args.lpips_start = 50001 # this doesn't do anything?
    
    #args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/sample"
    args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/train"
    args.test_data_dir = r"/home/wasabi/Documents/Projects/data/vae/test"
    args.checkpointing_steps = 5000# return to 5000
    args.report_to = 'wandb'

    #following are new parameters
    args.use_came = True
    args.diffusers_xformers = True
    args.save_for_SD = True
    args.save_precision = "fp16"
    args.train_only_decoder = True
    args.comment = "VAE finetune by Wasabi, test model using patched MSE"
    
    args.patch_loss = True
    args.patch_size = 64
    args.patch_stride = 32
    
    

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
    debug = True
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
    
    # string dtype:  accelerator.mixed_precision
    
    print("num process", accelerator.num_processes)
    print("working with", weight_dtype)

    save_for_SD = False
    if args.save_for_SD:
        save_for_SD = True
        save_dtype = get_dtype(args.save_precision)
        
    try: # keep it as float32, https://github.com/huggingface/diffusers/pull/6119
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
    # from the stackoverflow's answer, it links to the diffuser's sdxl training script example and in the code there's another link
    # which points to https://github.com/huggingface/diffusers/pull/6514#discussion_r1447020705
    # which may suggest we need to do all this casting before passing the learnable params to the optimizer
    if True:
        for param in vae.parameters():
            if param.requires_grad:
                # dtype conversion updated in place, updated to conversion code from gpt4
                #if accelerator.mixed_precision != "fp16":
                param.data.copy_(param.data.to(torch.float32))
                #else:
                #    param.data.copy_(param.data.to(torch.float16))
                #param.data.copy_(param.data.to(weight_dtype))
    else:
        vae = vae.half()
    vae = vae.half()
    # Load vae
    if args.use_ema:
        try:
            #ema_vae = AutoencoderKL.from_pretrained(
            #   args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float32)
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=weight_dtype)
        except:
            #ema_vae = AutoencoderKL.from_pretrained(
            #    args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32)
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)
        #ema_vae=ema_vae.half()
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
            vae.save_pretrained(os.path.join(output_dir, "vae_convert"), convert_vae_state_dict(vae.state_dict()))
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

                model_comment = ""
                if hasattr(args, "comment"):
                    model_comment = args.comment

                save_file(state_dict, os.path.join(output_dir, "vae_sdxl_ft.safetensors"), metadata={"comment":model_comment})

            # looking at ddpo_trainer by diffuser, it seems like it's a good idea to pop the weights as accelerator might do something
            # ensure accelerate doesn't try to handle saving of the model
            weights.pop()


        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), AutoencoderKL)
                ema_vae.load_state_dict(load_model.state_dict())
                ema_vae.to(accelerator.device, dtype=weight_dtype)
                del load_model

            # load diffusers style into model
            load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae", weight_dtype=weight_dtype)
            vae.register_to_config(**load_model.config)

            vae.load_state_dict(load_model.state_dict())
            

            # check comment in save_model_hook
            models.pop()
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        # Prepare everything with our `accelerator`.

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.encoder.to(accelerator.device, dtype=weight_dtype)
    vae.decoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    
    if args.train_only_decoder:
        # freeze the encoder weights
        for param in vae.encoder.parameters():
            param.requires_grad = False
        # set encoder to eval mode
        #vae.encoder.eval()
    
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
            v2.ToDtype(weight_dtype, scale=True),
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


    # we use a batch size of 1 bc we want to see samples side by side, which is made by the validation sample function
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn,
        batch_size=1, num_workers=1,  # args.train_batch_size*accelerator.num_processes,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    

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

    if args.patch_loss:
        patch_size = args.patch_size
        stride = args.patch_stride

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device, dtype=weight_dtype)
    lpips_loss_fn.requires_grad_(False)
    lpips_loss_fn.eval()  # added
    

    (
        vae, vae.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    ) = accelerator.prepare(
        vae, vae.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    lpips_loss_fn = accelerator.prepare(lpips_loss_fn)
    #if not args.train_only_decoder:
    vae.encoder = accelerator.prepare(vae.encoder)

    if args.use_ema:
        ema_vae.to(accelerator.device, dtype=weight_dtype)
        ema_vae = accelerator.prepare(ema_vae)

    for epoch in range(first_epoch, args.num_train_epochs):
        with torch.amp.autocast("cuda", dtype=torch.float16):#accelerator.autocast():
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
                    target = batch["pixel_values"]#.to(accelerator.device, dtype=weight_dtype)
                    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py

                    if accelerator.num_processes > 1:
                        posterior = vae.module.encode(target).latent_dist
                        z = posterior.sample().to(weight_dtype)
                        pred = vae.module.decode(z).sample.to(weight_dtype)
                    else:
                        posterior = vae.encode(target).latent_dist#.to(weight_dtype)
                        # z = mean                      if posterior.mode()
                        # z = mean + variable*epsilon   if posterior.sample()
                        z = posterior.sample().to(weight_dtype) # Not mode()
                        pred = vae.decode(z).sample.to(weight_dtype)

                    # pred = pred#.to(dtype=weight_dtype)
                    kl_loss = posterior.kl().mean().to(weight_dtype)

                    # mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                    
                    if args.patch_loss:
                        # patched loss
                        mse_loss = patch_based_mse_loss(target, pred, patch_size, stride).to(weight_dtype)
                        lpips_loss = patch_based_lpips_loss(lpips_loss_fn, target, pred, patch_size, stride).to(weight_dtype)

                    else:
                        # default loss
                        mse_loss = F.mse_loss(pred, target, reduction="mean").to(weight_dtype)
                        with torch.no_grad():
                            lpips_loss = lpips_loss_fn(pred, target).mean().to(weight_dtype)
                            if not torch.isfinite(lpips_loss):
                                lpips_loss = torch.tensor(0)
                                
                    if args.train_only_decoder:
                        # remove kl term from loss, bc when we only train the decoder, the latent is untouched
                        # and the kl loss describes the distribution of the latent
                        loss = (mse_loss 
                                + args.lpips_scale*lpips_loss)  # .to(weight_dtype)
                    else:
                        loss = (mse_loss 
                                + args.lpips_scale*lpips_loss 
                                + args.kl_scale*kl_loss)  # .to(weight_dtype)
                    if args.gradient_accumulation_steps and args.gradient_accumulation_steps>1:
                        loss = loss/args.gradient_accumulation_steps 

                    if not torch.isfinite(loss):
                        pred_mean = pred.mean()
                        target_mean = target.mean()
                        logger.info("\nWARNING: non-finite loss, ending training ")
                    
                    if debug and step < 10:
                        print(f"loss dtype: {loss.dtype}")
                        print(f"pred dtype: {pred.dtype}")
                        print(f"target dtype: {target.dtype}")
                        print(f"z dtype: {z.dtype}")
                        print(f"kl_loss dtype: {kl_loss.dtype}")
                        print(f"mse_loss dtype: {mse_loss.dtype}")
                        print(f"lpips_loss dtype: {lpips_loss.dtype}")
                        print(f"vae parameters dtype: {[param.dtype for param in vae.parameters()]}")


                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes

                # Gather the losses across all processes for logging (if we use distributed training).
                if loss is not None:
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps
                else:
                    logger.warning("Loss not defined, skipping gathering.")
                    
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
    
