# torch_cpu_loader.py

import torch

# Save the original torch.load function
original_torch_load = torch.load

# Define a new torch.load function that always loads to CPU
def torch_load_cpu(*args, **kwargs):
    return original_torch_load(*args, map_location=torch.device('cpu'), **kwargs)

# Override torch.load globally
torch.load = torch_load_cpu
