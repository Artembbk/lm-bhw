import numpy as np
import torch

def collate_fn(data):
    sequences = torch.stack([torch.from_numpy(pair[0]) for pair in data])
    lengths = torch.tensor([pair[1] for pair in data])
    return sequences, lengths

