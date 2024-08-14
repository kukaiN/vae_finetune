# vae_finetune
 code for finetuning VAE
 
 ### Comment by Wasabi:
 
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
- Relevant lib ver: torch==2.2.0+cu121, xformers==0.0.24, accelerate==0.25.0 
- The rest of libraries are listed in the requirements.txt (which is a ported requirement from a different project I'm working on)

Note: I didn't test on windows, but I also have a windows pc with an RTX3090 with pretty much same spec, so let me know if I need to test something on windows

 
### My modifications for testing:
- added CAME optimizer (similar to AdamW, but it uses confidence for estimating the steps, also lowers the VRAM overhead compared to base AdamW)
- use my own dataset (collection of anime images to see if it produces better smaller detailed images)

 