import torch
import torch.nn as nn
import torch.nn.functional as F


class CRESelfAttentionHead(nn.Module):
    """
    A single localized self-attention head to capture non-linear relationships
    between finance/legal terms (e.g., 'breach' vs 'maturity wall').
    Derived from Karpathy's nanoGPT design.
    """

    def __init__(self, embed_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length (tokens), Embedding dim
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities" between legal dark matter signals)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Causal masking
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        # Weighted aggregation of the financial contexts
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class CREDTSTransformerBlock(nn.Module):
    """
    Transformer layer that processes tokenized municipal litigation or title registries.
    """

    def __init__(self, embed_dim, n_head, block_size):
        super().__init__()
        head_size = embed_dim // n_head
        self.sa_heads = nn.ModuleList([CRESelfAttentionHead(embed_dim, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention + Residual Connection
        sa_out = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        x = x + self.ln1(self.proj(sa_out))
        # Feed Forward + Residual Connection
        x = x + self.ln2(self.ffn(x))
        return x


class CREDTS_v2_Model(nn.Module):
    """
    The complete CRE-DTS v2.0 Architecture.
    Bypasses standard Language Model Heads; directly maps text sequences to Survival Hazards.
    """

    def __init__(self, vocab_size, embed_dim, n_head, block_size, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

        # Stack of Karpathy-style Transformer blocks
        self.blocks = nn.Sequential(*[CREDTSTransformerBlock(embed_dim, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_dim)

        # --- THE SEVERE GENETIC MUTATION (The Survival Head) ---
        # Instead of projecting back to vocab_size (50257), we project to a single risk scalar h_theta(X)
        self.survival_head = nn.Linear(embed_dim, 1)

    def forward(self, idx):
        B, T = idx.shape

        # Ingest raw tokenized text chunks (e.g., encoded court filing scripts)
        tok_emb = self.token_embedding_table(idx)  # (B, T, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, embed_dim)
        x = tok_emb + pos_emb  # (B, T, embed_dim)

        # Process through Transformer layers to learn high-order cross-variable textual contexts
        x = self.blocks(x)  # (B, T, embed_dim)
        x = self.ln_f(x)  # (B, T, embed_dim)

        # Global pooling over the time/token dimension (Average Pooling)
        pooled_context = torch.mean(x, dim=1)  # (B, embed_dim)

        # Output the continuous log-hazard ratio for DeepSurv tracking
        log_hazard_ratio = self.survival_head(pooled_context)  # (B, 1)
        return log_hazard_ratio


# --- Demonstration of Training Setup with DeepSurv Loss ---
if __name__ == "__main__":
    # Hyperparameters for sub-scale MVP training (nanoGPT size)
    vocab_size = 5000  # Token dictionary size for real estate law domain
    block_size = 256  # Context window length (max sequence length of a court docket summary)
    embed_dim = 64
    n_head = 4
    n_layer = 3
    batch_size = 8

    # Initialize the v2.0 Model
    model = CREDTS_v2_Model(vocab_size, embed_dim, n_head, block_size, n_layer)
    print("CRE-DTS v2.0 Model Initialized successfully.")

    # Simulated input: 8 different court documents tokenized into sequences of length 256
    simulated_text_batch = torch.randint(0, vocab_size, (batch_size, block_size))

    # Execute forward pass through Transformer layers to extract hidden default hazard metrics
    predicted_risk_scalars = model(simulated_text_batch)
    print(f"Output Shape (Batch, Risk Scalar): {predicted_risk_scalars.shape} -> 1 scalar per document.")