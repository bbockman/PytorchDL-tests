import torch
import torch.nn as nn
import torch.nn.functional as F

# note, beyond custom self-attention, we need absolute or relative positional encodings pi_i,j or PI
# we also need mlp to flatten extend and contract back, between attention layers, merging information
# finally layernorms, rather than batchnorms are standard between layers
# x = layernorm(x + mhsa(x)), x = layernorm(xn + mlp(xn)) constitutes standard transformer block
# positional encodings can be applied relative, by term-by-term multiplication with attention matrix etc.
# in nlp we also have tokenization step, then Omega_nu(vocab -> embed) * T (one hot) -> X
# example tokenization would be word piece tokenization, byte pair encoding, etc.
# note that mlp is generally FC layer per token done in parallel, should confirm how these are generally recombined
# conv doesn't seem off-limits either depending on application.
# finally, mhsa / mlp steps are generally combined with residual connections

# Finally, the embedding matrix X representing the text is passed through a
# series of K transformers, called a transformer model. There are three types
# of transformer models. An encoder transforms the text embeddings into a
# representation that can support a variety of tasks. A decoder predicts the
# next token to continue the input text. Encoder-decoders are used in
# sequence-to-sequence tasks, where one text string is converted into another
# (e.g., machine translation). These variations are described in sections 12.6–
# 12.8, respectively.

# Note in case of iterative generative applications, nucleus sampling appears most promising
# out of the beam-search / top-k / nucleus sampling options for the iterative choosing process

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, N, embed_dim)  batch_size, sequence_length, embedding_dim
        """
        B, N, _ = x.shape

        # 1) Linear projections
        Q = self.W_q(x)  # (B, N, embed_dim)
        K = self.W_k(x)  # (B, N, embed_dim)
        V = self.W_v(x)  # (B, N, embed_dim)

        # 2) Split into heads
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4) Weighted sum
        context = torch.matmul(attn_probs, V)  # (B, num_heads, N, head_dim)

        # 5) Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)  # (B, N, embed_dim)

        # 6) Output projection
        out = self.W_o(context)  # (B, N, embed_dim)
        return out
