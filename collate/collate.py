import numpy as np
import torch

def collate_fn(data):
    arrays, lengths = data
    tensor_list = [torch.from_numpy(arr).to(torch.long) for arr in arrays]

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return tensor_list, lengths_tensor