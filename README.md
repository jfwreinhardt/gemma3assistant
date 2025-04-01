# gemma3assistant

This Python script lets you run Gemma 3 locally as your private AI assistant. No GPU is required.  Once installed, no internet connection is required.

Tested on Linux and Windows using Python 3.10 or Python 3.11

To install:
1. Setup your Python environment by installing the following packages:
pip install transformers==4.50.3 accelerate==1.5.2 tokenizers==0.21.1 torch==2.1.2 torchvision -U

2. Download the "gemma-3-1b-it" model as tar.gz from kaggle
https://www.kaggle.com/models/google/gemma-3/transformers/gemma-3-1b-it

3. Unpack the tar.gz file from step 2 into "C:\gemma3_1b_it_model".  You should now see a model.safetensors file and several other smaller files in this directory.

To run:
python gemma3assistant.py

![gemma3assistant screenshot](https://github.com/user-attachments/assets/aa3f4a7e-bfb7-4618-9dad-8b513e904b56)
