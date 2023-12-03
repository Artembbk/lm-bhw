import torch

def create_non_special_mask(lengths, max_length):
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length)
    for i, length in enumerate(lengths):
        mask[i, 1:length] = 1
    return mask