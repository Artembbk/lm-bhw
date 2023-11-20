import torch
import torch.nn as nn

class PrenormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(PrenormTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(torch.relu(self.linear1(src2)))
        src = src + self.dropout2(src2)

        return src

class PrenormTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(PrenormTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, lengths):
        max_length = src.size(1)
        
        print(torch.arange(max_length).expand(len(src), max_length).shape, lengths.unsqueeze(1).shape)
        mask = torch.arange(max_length).expand(len(src), max_length) >= lengths.unsqueeze(1)
        padding_mask = mask.unsqueeze(1).unsqueeze(2).to(src.device)
        look_ahead_mask = ~torch.triu(torch.ones(max_length, max_length)).unsqueeze(0).bool()

        for layer in self.layers:
            src = layer(src, look_ahead_mask, padding_mask)
        return src
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('positional_encoding', self.get_positional_encoding(d_model, max_len))

    def get_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1)].detach()
        return x

class BaseModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super(BaseModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = PrenormTransformerEncoder(
            PrenormTransformerEncoderLayer(d_model, nhead), num_layers)
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded_with_position = self.positional_encoding(embedded)
        encoded = self.transformer_encoder(embedded_with_position, lengths)
        next_token_prediction = self.linear(encoded)
        return next_token_prediction
