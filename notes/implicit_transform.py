import torch
import torch.nn.functional as F


class ImplicitQueryAttention(torch.nn.Module):
    def __init__(self, embed_dim, n_iters=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_iters = n_iters

        # Output-dependent query projection
        self.Omega_Q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        # Key projection from input
        self.W_K = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len, embed_dim)
        """
        K = self.W_K(x)  # (B, N, D)
        y = x.clone()  # initial guess for output
        for _ in range(self.n_iters):
            Q = self.Omega_Q(y)  # query depends on output
            attn_scores = torch.matmul(K, Q.transpose(-2, -1))  # (B, N, N)
            attn_probs = F.softmax(attn_scores, dim=-1)
            y = torch.matmul(attn_probs, x)  # values are just x
        return y


B, N, D = 2, 4, 8  # batch, sequence length, embedding dim
x = torch.randn(B, N, D)
layer = ImplicitQueryAttention(D, n_iters=20)
y = layer(x)
print(y.shape)  # (B, N, D)
