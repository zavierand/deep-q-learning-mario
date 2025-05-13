import torch

### Check if CUDA is available
print(torch.version.cuda)   # Should return the version of CUDA
print(torch.cuda.is_available())  # Should return True if CUDA is working
print(torch.cuda.device_count())  # Should return the number of GPUs available  
print(torch.cuda.current_device())  # Should return the index of the current device
print(torch.cuda.get_device_name(0))  # Should return the name of the GPU