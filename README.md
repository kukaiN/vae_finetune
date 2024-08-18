# vae_finetune
 code for finetuning VAE
 
 ### Comment by Wasabi:
 
 How to run code:
 ```
 (start venv)
 accelerate launch --num_cpu_threads_per_process=1 vae_finetune.py
 ```
 
 code was initially copied from the comment from the diffuser's github issue: https://github.com/huggingface/diffusers/issues/3726
 
 In the github issue, there's people having problems running the code with mixed precision (fp16 and bf16). I also faced this mixed precision issue so I made some modifications that made it work. I'll list the solution to fixing the mixed precision.
 
 Although it doesn't explicitly state it in the error message, but the following are also resolved with the following modifications, as the errors are pertaining to mixed precisions:
 - "Trying to create tensor with negative dimension -17146298352"
 - "RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same"
 
 Also I plan on making this a separate "fork" so I can add my tweaks here and there. I'll list the modifications I do separate from the minimum required changes needed to fix the mixed precision.
 
 ### modifications for fixing mixed precision:
 - I changed the order of some of the code so weight_dtype is known before anything
 - Iterate through the trainable parameters and initialize them with torch.float32, then let mixed precision convert it automatically
 - added "with accelertor.autocast():" to autocast the inputs from the encoder all the way to the accelerator.backward(loss)
 
### My (secondary) PC specs:
- Ubuntu 24.04
- Python 3.10
- RTX3090 (with cuda + cudnn)
- Relevant lib ver: torch==2.4.0+cu121, xformers==0.0.27.post2, accelerate==0.33.0 
- The rest of libraries are listed in the requirements.txt (which is a ported requirement from a different project I'm working on)

Note: I didn't test on windows, but I also have a windows pc with an RTX3090 with pretty much same spec, so let me know if I need to test something on windows

 
### My modifications for testing:
- added xformers (i didn't transfer the case when xformer is disabled so I'll update this within a few days
- added CAME optimizer fom pytorch_optimizer (similar to AdamW, but it uses confidence for estimating the steps, also lowers the VRAM overhead compared to base AdamW)
- using my own dataset (collection of anime images to see if it produces better smaller detailed images)
- added accelerator's unwrap_model function directly cause it's causing problems for higher diffuser and accelerate versions. (accelerator.unwrap_model(vae) --> acc_unwrap_model(vae))
 
