import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    """
    Recursively initialize weights for Transformers.
    Can be used with model.apply(init_weights).
    Handles Linear, Conv2d, LayerNorm, and top-level cls_token / pos_embed.
    """
    # Linear layers (MLP and attention projections)
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # Convolutional layers (for patch embedding or conv stem)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # LayerNorm
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # Top-level Transformer-specific parameters
    # Only initialize if the module has these attributes
    if hasattr(module, 'cls_token'):
        nn.init.trunc_normal_(module.cls_token, std=0.02)
    if hasattr(module, 'pos_embed'):
        nn.init.trunc_normal_(module.pos_embed, std=0.02)

# -----------------------------
# Transformer Block (your style)
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------
# Patch Embedding for MNIST
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x


class LinearPatch(nn.Module):
    """
    Safe manual patch embedding (no conv).
    Input:  x (B, C, H, W) where H and W are divisible by patch_size.
    Output: (B, n_patches, embed_dim)
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        returns: (B, n_patches, embed_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        H_blocks = H // p
        W_blocks = W // p
        n_patches = H_blocks * W_blocks

        # reshape into (B, C, H_blocks, p, W_blocks, p)
        # note: use contiguous() to be safe before view/reshape
        x = x.contiguous().view(B, C, H_blocks, p, W_blocks, p)

        # reorder to (B, H_blocks, W_blocks, C, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # collapse to (B, n_patches, C * p * p)
        x = x.view(B, n_patches, C * p * p)

        # project each flattened patch to embed_dim
        x = self.proj(x)  # (B, n_patches, embed_dim)

        # update attribute (useful externally)
        self.n_patches = n_patches

        return x



# -----------------------------
# Full Transformer Model
# -----------------------------
class MNISTTransBase(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64,
                 depth=4, n_heads=4, mlp_ratio=4.0, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        self.apply(init_weights)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)

        # prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # take cls token
        out = self.head(cls_out)
        return out


class MNISTTransRef(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64,
                 depth=4, n_heads=4, mlp_ratio=4.0, num_classes=10, dropout=0.1):
        super().__init__()
        # Patch embedding
        self.patch_embed = LinearPatch(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stack multiple transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Final normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        self.apply(init_weights)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Sequentially pass through all Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out


# -----------------------------
# Convolutional stem
# -----------------------------
class ConvStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)  # reduce 28x28 → 14x14
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x  # (B, C, 14, 14)

# -----------------------------
# Hybrid CNN + Transformer
# -----------------------------
class MNIST_HybridTransformer(nn.Module):
    def __init__(self, num_classes=10,
                 conv_out_channels=64,
                 embed_dim=128,
                 depth=12,
                 n_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()

        # 1) Convolutional stem
        self.stem = ConvStem(in_channels=1, out_channels=conv_out_channels)

        # 2) Patch embedding
        self.patch_embed = PatchEmbedding(in_channels=conv_out_channels, embed_dim=embed_dim, patch_size=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 3) Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # 4) Final normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 5) Initialization
        self._init_weights()

    def forward(self, x):
        B = x.size(0)

        # 1) Conv stem
        x = self.stem(x)  # (B, C, 14, 14)

        # 2) Patch embedding
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)

        # 3) Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4) Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) Final normalization and classifier
        x = self.norm(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out
