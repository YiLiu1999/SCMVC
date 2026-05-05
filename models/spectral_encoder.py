import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        d_k = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_k)

        q = q.transpose(1, 2)                    # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                    # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)   # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention: softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_k)
        x = self.output_layer(x).squeeze()
        return x


class SpectralTransformer(nn.Module):
    """Transformer encoder for spectral feature extraction."""

    def __init__(self, hidden_size, ffn_size, feature_dim, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.relu = nn.ReLU()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_size, feature_dim)
        self.projector = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, attn_bias=None):
        x = x.to(torch.float32)

        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        feature = self.fc(x)
        projection = self.projector(self.relu(feature))
        return F.normalize(feature, dim=-1), F.normalize(projection, dim=-1)
