import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ImplicitQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_iters=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_iters = n_iters
        self.Omega_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, track=False):
        K = self.W_K(x)
        y = x.clone()
        y_history = [y.clone()] if track else None

        for _ in range(self.n_iters):
            Q = self.Omega_Q(y)
            attn_scores = torch.matmul(K, Q.transpose(-2, -1))
            attn_probs = F.softmax(attn_scores, dim=-1)
            y = torch.matmul(attn_probs, x)
            if track:
                y_history.append(y.clone())
        return (y, y_history) if track else y

# Example
B, N, D = 1, 4, 8
x = torch.randn(B, N, D, requires_grad=True)
layer = ImplicitQueryAttention(D, n_iters=10)

y, y_hist = layer(x, track=True)

# Compute norm of difference between iterations
diffs = [torch.norm(y_hist[i+1] - y_hist[i]).item() for i in range(len(y_hist)-1)]
plt.plot(diffs, marker='o')
plt.xlabel("Iteration")
plt.ylabel("||y_{t+1} - y_t||")
plt.title("Convergence of Fixed-Point Iteration")
plt.show()