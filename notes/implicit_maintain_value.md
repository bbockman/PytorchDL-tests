# Implicit Query Attention with Linear Values

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ImplicitQueryAttentionLinearV(nn.Module):
    def __init__(self, embed_dim, n_iters=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_iters = n_iters
        # Linear transforms
        self.Omega_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Omega_V = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, track=False):
        """
        x: (B, N, D) input
        track: bool, store intermediate y for visualization
        Returns:
            y: converged output
            y_history: list of y per iteration if track=True
        """
        K = self.W_K(x)          # Keys from input
        V = self.Omega_V(x)      # Values from input
        y = V.clone()            # Initialize y with V
        y_history = [y.clone()] if track else None

        for _ in range(self.n_iters):
            Q = self.Omega_Q(y)                     # Queries depend on output y
            attn_scores = torch.matmul(K, Q.transpose(-2, -1)) / (self.embed_dim ** 0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            y = torch.matmul(attn_probs, V)         # Apply linear values
            if track:
                y_history.append(y.clone())

        return (y, y_history) if track else y

# Example usage
B, N, D = 1, 4, 8
x = torch.randn(B, N, D, requires_grad=True)
layer = ImplicitQueryAttentionLinearV(D, n_iters=10)

y, y_hist = layer(x, track=True)

# Compute convergence metric (difference norm between iterations)
diffs = [torch.norm(y_hist[i+1] - y_hist[i]).item() for i in range(len(y_hist)-1)]
plt.plot(diffs, marker='o')
plt.xlabel("Iteration")
plt.ylabel("||y_{t+1} - y_t||")
plt.title("Convergence of Fixed-Point Iteration with Linear V")
plt.show()

# Optional: backward pass through final y
loss = y.sum()
loss.backward()
print("Gradient w.r.t input x:", x.grad)
```

### Explanation

1. **Keys (`K`)**: Always computed from input `x`.
2. **Values (`V`)**: Now a linear transform of `x` (`Omega_V x`) instead of just `x`.
3. **Queries (`Q`)**: Implicitly depend on the current output estimate `y`.
4. **Fixed-point iteration**: Updates `y` using attention on `(K, Q)` and `V`.
5. **Visualization**: Tracks `||y_{t+1} - y_t||` to monitor convergence.
6. **Backward pass**: Standard PyTorch autograd handles unrolled iterations.

This is fully general and allows you to experiment with implicit query attention **while maintaining a trainable value transform**.  

---

