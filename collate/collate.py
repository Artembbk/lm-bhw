import numpy as np
import torch

def collate_fn(arrays, numbers):
    tensor_list = [torch.from_numpy(arr).to(torch.long) for arr in arrays]

    numbers_tensor = torch.tensor(numbers, dtype=torch.long)
    
    return tensor_list, numbers_tensor