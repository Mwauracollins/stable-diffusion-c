import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention

class CLIPMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
    
        self.linear_1 = nn.Linear(d_model, 4 * d_model)
        self.linear_2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        return x
    

class CLIPBlock(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttention(config.n_heads, config.d_model)
        self.mlp = CLIPMLP(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.wpe = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.blocks = nn.ModuleList(
            [CLIPBlock(self.config) for _ in range(self.config.n_layers)]
            )
        self.ln_final = nn.LayerNorm(self.config.d_model)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.config.max_seq_len, "Cannot forward, model not large enough"

        pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0).to(x.device) # Shape: (1, T)
        x = self.wte(x) + self.wpe(pos) # Shape: (B, T, d_model)

        for block in self.blocks:
            x = block(x)
        output = self.ln_final(x)

        return output