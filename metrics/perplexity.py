import torch
from torch.nn import functional as F

def perplexity(input_ix, logits, mask):
    """
    :param 
        model: language model that can compute next token logits given token indices
        input_ix: int32 matrix of tokens, shape: [batch_size, length]; padded with eos_ix
        mask: mask of non-special tokens, shape: [batch_size, length]
    
    :returns: scalar perplexity, mean perplexity over non-eos tokens
    """
    logits = logits[:, :-1, :]  # Remove the last token predictions

    # Flatten the logits and input_ix
    logits = logits.contiguous().view(-1, logits.size(-1))
    flat_input_ix = input_ix[:, 1:].contiguous().view(-1)  # Remove the first token (EOS) and flatten

    ce = F.cross_entropy(logits, flat_input_ix, reduction='none')
    masked_ce = ce * mask[:, 1:].contiguous().view(-1).float()  # Apply mask

    total_loss = masked_ce.sum()
    total_non_eos = mask[:, 1:].sum()

    perplexity = torch.exp(total_loss / total_non_eos)
    return perplexity.item()  # Return the scalar value as a Python float