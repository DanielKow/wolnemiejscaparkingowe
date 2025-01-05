import torch

print(torch.__version__)

if (torch.backends.mps.is_available()):
    print("MPS is available!")

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available on this system.")