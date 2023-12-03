from torch import nn
import torch

class MultiHeadSelfAttention(nn.Module):
    """
    Implemetn MultiHeadSelfAttention from the article https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, n_heads, max_seq_length, dropout_p=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        
        # initialize key, query, value matrix in one batch for optimization
        self.QKV = nn.Linear(hidden_size, 3*hidden_size)
        
        # output projection
        self.projector =  nn.Linear(hidden_size, hidden_size)
        
        # regularization using dropout
        self.attn_dropout =  nn.Dropout(dropout_p)
        self.proj_dropout =  nn.Dropout(dropout_p)
        
        # create causal mask to ensure that attention is only applied to the left in the input sequence
        # causal_mask is not a part of model weights so that we have to use register_buffer method
        mask_weights =  torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer("causal_mask", mask_weights) # shape = [1, 1, max_seq_length, max_seq_length]

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size() # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and split batch into three parts for q, k, v
        # move head forward to be the batch dim

        # Reshape [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, self.n_head, hidden_size // self.n_head]
        # Transpose [batch_size, seq_len, self.n_head, hidden_size // self.n_head] -> [batch_size, self.n_head, seq_len, hidden_size // self.n_head]
        # in order to calculate attention over different heads

        qkv = self.QKV(x).view(batch_size, seq_len, self.n_heads, 3, -1)
        q, k, v = qkv.split(1, dim=-2)
        q = q.squeeze(-2)
        k = k.squeeze(-2)
        v = v.squeeze(-2)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        
        attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        
        y = self.projector(attn_output)
        y = self.proj_dropout(y)
        return y
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_length, dropout_p=0.1):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(hidden_size, n_heads, max_seq_length, dropout_p)
        

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),  # Adjust this multiplier as needed
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_p)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        

    def forward(self, x):
        attn_output = self.self_attention(self.layer_norm1(x) + x)

        # Residual connection and layer normalization
        x = self.layer_norm2(attn_output + x)

        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_length=32, hidden_size=16, n_layer=2, n_heads=2, dropout_p=0.1):
        super().__init__()
        
        self.max_seq_length = max_seq_length
        # Initialize main gpt blocks: embedding layer, positional embedding layer, dropout, 
        # transformer blocks, layer norm, linear head for projection from hidden size to vocab size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.layers = nn.Sequential(*[TransformerBlock(hidden_size, n_heads, max_seq_length, dropout_p) for _ in range(n_layer)])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, input_ix):
        batch_size, seq_len = input_ix.size()
        assert seq_len <= self.max_seq_length, f"Cannot forward sequence of length {seq_len}, block size is only {self.max_seq_length}"

        # Create position embeddings
        # YOUR CODE
        
        position_ids = torch.arange(seq_len, device=input_ix.device).expand(batch_size, seq_len)
        position_embeddings = self.positional_embedding(position_ids)

        # Create token's embeddings
        # YOUR CODE
        
        x = self.embedding(input_ix)
        x += position_embeddings
        x = self.dropout(x)
        
        x = self.layers(x)
        
        x = self.layer_norm(x)
        
        x = self.fc(x)
        
        return x