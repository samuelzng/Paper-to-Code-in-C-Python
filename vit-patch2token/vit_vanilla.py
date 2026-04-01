import torch
import torch.nn as nn
import copy
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=28, patch_size=7, in_channels=1, d_model=64):
        super().__init__()
        self.patch_size = patch_size

        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, d_model)

        # Learnable [CLS] token — 用于最终分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding covers [CLS] + all patches
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, d_model))

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract patches via unfold
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B, C, H', W', p, p]
        x = x.contiguous().view(B, -1, C * p * p)       # [B, num_patches, patch_dim]
        x = self.proj(x)                                 # [B, num_patches, d_model]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)            # [B, 1 + num_patches, d_model]

        return x + self.pos_embed                         # [B, 1 + num_patches, d_model]


def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot-Product Attention"""
    # Q, K, V -> [B, h, seq_len, d_k]
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Wq, Wk, Wv, Wo
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Input: [B, seq_len, d_model]
        B = query.size(0)

        # Project and reshape: [B, seq_len, d_model] -> [B, h, seq_len, d_k]
        query, key, value = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]

        # Scaled dot-product attention
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # Concat heads: [B, h, seq_len, d_k] -> [B, seq_len, d_model]
        x = (
            x.transpose(1, 2)                          # FIX: was (-1, -2), should be (1, 2)
            .contiguous()
            .view(B, -1, self.d_k * self.h)
        )

        # Output projection — FIX: was missing in original
        return self.linears[3](x)


class PointwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN)"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))


class SublayerConnection(nn.Module):
    """Pre-Norm residual connection: x + Sublayer(LayerNorm(x))"""
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)    # Final norm after all layers

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ViTClassifier(nn.Module):
    """
    Vision Transformer for image classification.
    Uses [CLS] token for classification (standard ViT approach).
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 d_model=64, d_ff=256, h=4, N=4, dropout=0.1, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)

        # Classification head on [CLS] token output
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)             # [B, 1+num_patches, d_model]
        x = self.encoder(x, mask=None)      # [B, 1+num_patches, d_model]
        cls_output = x[:, 0]                # [B, d_model]  — take [CLS] token
        return self.classifier(cls_output)  # [B, num_classes]


if __name__ == "__main__":
    model = ViTClassifier()
    dummy = torch.randn(4, 1, 28, 28)       
    logits = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {logits.shape}")          # Expected: [4, 10]
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")