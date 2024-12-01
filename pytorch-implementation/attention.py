import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, x):
        B, T, C = x.shape

        # Project input to Q, K, V matrices
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3 * d_model)
        qkv = qkv.view(B, T, self.n_heads, 3 * self.d_k)  # Reshape to (B, T, n_heads, 3 * d_k)
        Q, K, V = qkv.chunk(3, dim=-1)  # Split last dimension into Q, K, V

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # Shape: (B, n_heads, T, d_k)
        K = K.transpose(1, 2)  # Shape: (B, n_heads, T, d_k)
        V = V.transpose(1, 2)  # Shape: (B, n_heads, T, d_k)

        if x.is_cuda and torch.cuda.is_available():
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)
        else:
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5  # Scaled dot-product
            attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax over last dim
            out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        out = self.out_proj(out)

        return out

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, x, enc):
        B, T, C = x.shape  # Query shape: [Batch, Target_len, C]
        _, T_enc, _ = enc.shape  # Encoder output shape: [Batch, T_encoder, C]

        # 1. Project queries, keys, and values
        Q = self.q_proj(x)  # Shape: [B, T, C]
        K = self.k_proj(enc)  # Shape: [B, S, C]
        V = self.v_proj(enc)  # Shape: [B, S, C]

        # 2. Reshape into multiple heads and prepare for attention computation
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # Shape: [B, n_heads, T, d_k]
        K = K.view(B, T_enc, self.n_heads, self.d_k).transpose(1, 2)  # Shape: [B, n_heads, S, d_k]
        V = V.view(B, T_enc, self.n_heads, self.d_k).transpose(1, 2)  # Shape: [B, n_heads, S, d_k]

        # 3. Compute scaled dot-product attention
        if x.is_cuda and torch.cuda.is_available():
            attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)
        else:
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)  # Shape: [B, n_heads, T, S]
            attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: [B, n_heads, T, S]
            attn_output = torch.matmul(attn_weights, V)  # Shape: [B, n_heads, T, d_k]

        # 4. Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # Shape: [B, T, n_heads, d_k]
        attn_output = attn_output.view(B, T, C)  # Shape: [B, T, C]

        # 5. Output projection
        output = self.out_proj(attn_output)  # Shape: [B, T, C]

        return output