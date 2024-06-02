import torch

print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version: ", torch.version.cuda)
    print("cuDNN version: ", torch.backends.cudnn.version())
    print("Device name: ", torch.cuda.get_device_name(0))
    print("Number of available devices: ", torch.cuda.device_count())
