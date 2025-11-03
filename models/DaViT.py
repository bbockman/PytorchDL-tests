import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------
# Utilities
# ------------------------------
def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std)

def init_weights(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    # top-level params (cls_token / pos_embed) are set by model after apply

# ------------------------------
# Patch embedding (manual)
# ------------------------------
class LinearPatch(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.H_blocks = img_size // patch_size
        self.W_blocks = img_size // patch_size
        self.n_patches = self.H_blocks * self.W_blocks

        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        assert H == W
        assert H == self.img_size
        assert C == self.in_channels

        p = self.patch_size
        n_patches = self.n_patches
        patch_dim = self.patch_dim

        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, n_patches, patch_dim)

        x = self.proj(x)  # (B, n_patches, embed_dim)
        return x

# ------------------------------
# Multi-head self-attention (generic)
# operates on last dimension as embedding, and sequence length is axis -2
# input: (B, L, E) -> output: (B, L, E)
# ------------------------------
class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, L, E)
        B, L, E = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # (B, L, H, head_dim) -> transpose -> (B, H, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if mask is not None:
            attn = attn + mask  # mask should be additive
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, E)  # (B, L, E)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

# ------------------------------
# MHSA variant for channel attention:
# We will reuse MHSA by transposing input to (B, L_ch, E_ch)
# For channel-attention we will supply xT with shape (B, D, N), and use embed_dim = N
# So we construct a separate MHSA with embed_dim = N and heads dividing N.
# ------------------------------

# ------------------------------
# Transformer MLP
# ------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# ------------------------------
# DaViT Block: Spatial attention (tokens) then Channel attention (channels)
# Each with its own LayerNorm and small MLP
# ------------------------------
class DaViTBlock(nn.Module):
    def __init__(self, n_tokens, embed_dim,
                 spatial_heads=8, channel_heads=5,
                 mlp_ratio=4.0, dropout=0.0):
        """
        n_tokens: number of tokens (patches + cls)
        embed_dim: D
        spatial_heads divides embed_dim
        channel_heads divides n_tokens
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

        # Spatial attention operates on (B, n_tokens, D)
        self.norm_sp = nn.LayerNorm(embed_dim)
        self.s_attn = MHSA(embed_dim, spatial_heads, dropout=dropout)
        self.mlp_sp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

        # Channel attention: operate on channels as sequence length (D tokens)
        # We'll build an MHSA where embed_dim_for_channel = n_tokens (so heads divide n_tokens)
        assert n_tokens % channel_heads == 0, "n_tokens must be divisible by channel_heads for channel MHSA"
        self.norm_ch = nn.LayerNorm(embed_dim)  # applied before transpose
        self.channel_mhsa = MHSA(n_tokens, channel_heads, dropout=dropout)
        # After channel-attention we mix back with an MLP in feature space
        self.mlp_ch = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        """
        x: (B, n_tokens, D)
        returns: (B, n_tokens, D)
        """
        # --- Spatial attention (token-wise) ---
        x_sp = x
        x = x + self.s_attn(self.norm_sp(x))  # residual
        x = x + self.mlp_sp(self.norm_sp(x))  # another residual using same norm (you can use separate)

        # --- Channel attention ---
        # We want to let channels (D) attend, using each channel's vector of length n_tokens.
        # Transpose to (B, D, n_tokens) so sequence length is D and embedding per token is n_tokens.
        x_norm = self.norm_ch(x)  # (B, n_tokens, D)
        xT = x_norm.transpose(1, 2).contiguous()  # (B, D, n_tokens)
        # but MHSA expects (B, L, E) with last dim = embed_dim; here embed_dim = n_tokens
        # so we treat L=D, E=n_tokens: reshape to (B, D, n_tokens) already correct
        ch_out = self.channel_mhsa(xT)  # input shape (B, L=D, E=n_tokens) -> output same shape
        # transpose back
        ch_out = ch_out.transpose(1, 2).contiguous()  # (B, n_tokens, D)
        # residual + MLP in feature space
        x = x + ch_out
        x = x + self.mlp_ch(self.norm_ch(x))
        return x

class PaddedPatch(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64, padding=2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = ( (img_size + padding*2) // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

from models.mnist_transformers import PatchEmbedding
# ------------------------------
# DaViT model for MNIST classification
# ------------------------------
class DaViT_MNIST(nn.Module):
    def __init__(self,
                 img_size=28, patch_size=4, in_channels=1,
                 embed_dim=64, depth=8,
                 spatial_heads=4, channel_heads=5,
                 mlp_ratio=4.0, dropout=0.1,
                 num_classes=10):
        super().__init__()
        # patch embedding
        self.patch_embed = LinearPatch(img_size=img_size, patch_size=patch_size,
                                       in_channels=in_channels, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding for tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Build blocks
        self.blocks = nn.ModuleList([
            DaViTBlock(n_tokens=n_patches + 1,
                       embed_dim=embed_dim,
                       spatial_heads=spatial_heads,
                       channel_heads=channel_heads,
                       mlp_ratio=mlp_ratio,
                       dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # initialization
        self.apply(init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, 1, H, W)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, n_patches, D)
        # prepend cls token
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        logits = self.head(cls_out)  # (B, num_classes)
        return logits

# ------------------------------
# Smoke test
# ------------------------------
if __name__ == "__main__":
    B = 8
    model = DaViT_MNIST(img_size=28, patch_size=4, embed_dim=128, depth=6,
                        spatial_heads=8, channel_heads=5, mlp_ratio=4.0, dropout=0.1)
    dummy = torch.randn(B, 1, 28, 28)
    out = model(dummy)
    print("out", out.shape)   # (B, 10)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)
