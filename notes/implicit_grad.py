import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_iters=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_iters = n_iters

        # Output-dependent query projection
        self.Omega_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        # Key projection from input
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len, embed_dim)
        """
        K = self.W_K(x)  # Keys (B, N, D)
        y = x.clone()  # Initialize output guess
        for _ in range(self.n_iters):
            Q = self.Omega_Q(y)  # Query depends on output
            attn_scores = torch.matmul(K, Q.transpose(-2, -1))  # (B, N, N)
            attn_probs = F.softmax(attn_scores, dim=-1)
            y = torch.matmul(attn_probs, x)  # Values = x
        return y


# --- Simple test with gradients ---
B, N, D = 2, 4, 8
x = torch.randn(B, N, D, requires_grad=True)

layer = ImplicitQueryAttention(D, n_iters=5)
y = layer(x)

# Define a simple loss: sum of outputs
loss = y.sum()
loss.backward()

# Print gradients for inspection
print("Gradients w.r.t input x:")
print(x.grad)

print("\nGradients w.r.t Omega_Q parameters:")
for name, param in layer.named_parameters():
    if param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm().item():.4f}")
