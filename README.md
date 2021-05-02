# About
Reinforcement learning-based algorithmic trading

# Installation
Run the following to setup dependencies, preferably using a virtualenv to avoid issues:
```bash
pip install -r requirements.txt
```
This project should be compatible with Python 3.6-3.9.


# for training using GPU, if encounter following problem:

## related to cusolver64_10.dll
Step 1

Move to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin

Step 2

Rename file cusolver64_11.dll  To  cusolver64_10.dll

for further detial https://stackoverflow.com/questions/65608713/tensorflow-gpu-could-not-load-dynamic-library-cusolver64-10-dll-dlerror-cuso

## related to cudnn64_8.dll

you probably need to install cudnn, for detail see

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download-windows
