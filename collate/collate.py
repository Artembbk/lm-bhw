import numpy as np
import torch

def collate_fn(data):
    sequences = [torch.tensor(pair[0], dtype=torch.int64) for pair in data]
    sequences = torch.cat(sequences)
    print(sequences.shape)
    lengths = torch.tensor([pair[1] for pair in data])
    return sequences, lengths

