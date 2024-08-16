#!  /bin/bash

echo "Installing Requirements on a Linux platform"
echo "Make sure to have python3, recomended 3.10 or higher"

# create venv
python3 -m venv .venv

# activate venv
source .venv/bin/activate

# check for if we're running inside a venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
  INVENV=1
else
  INVENV=0
fi

# if we're in a venv install the requirements
if [[ $INVENV -eq 1 ]]; then
  pip install --upgrade pip
  
  pip install --upgrade -r linux_requirements.txt
  
  # for linux, need additional wheel to get xformers
  pip install wheel
  #pip install xformers==0.0.24
  # base torch and xformers are installed in the requirements.txt and we upgrade it to the cuda version here
  # we need to install it later bc installing xformers overwrites it to the base torch
  pip install --upgrade torch>=2.2.0 torchvision torchaudio xformers>=0.0.24 --index-url https://download.pytorch.org/whl/cu121
  #pip install git+https://github.com/openai/CLIP.git
else
  echo "failed to start virtualenv, do you have python3.xx-venv?"
fi

