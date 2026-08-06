import torch

print(torch.cuda.is_available()) # devrait être True
print(torch.cuda.get_device_name(0)) # devrait afficher "NVIDIA GeForce RTX 5070 Laptop"
print(torch.version.cuda)
