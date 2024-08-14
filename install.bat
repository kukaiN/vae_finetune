@echo off

python -m venv .venv

call .venv\Scripts\activate

pip install -r requirements.txt


pip install --upgrade torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM pip install git+https://github.com/openai/CLIP.git

pause
