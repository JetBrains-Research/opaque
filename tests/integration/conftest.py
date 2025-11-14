"""Shared fixtures for integration tests."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model Definitions
# ============================================================================


class TinyLLaMA(nn.Module):
    """
    Minimal LLaMA-like causal language model.

    Architecture:
    - Token embeddings
    - N transformer blocks (custom implementation using nn.MultiheadAttention)
    - Output head

    Note: We use custom TransformerBlock instead of nn.TransformerDecoderLayer
    because PyTorch's built-in transformer layers have issues with functional_call.
    """

    def __init__(self, vocab_size=1000, dim=128, n_layers=2, n_heads=4, max_seq_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])

        # Output
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        """
        Args:
            tokens: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # Embed tokens
        h = self.tok_embeddings(tokens)  # [batch, seq_len, dim]

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Output projection
        h = self.norm(h)
        logits = self.output(h)  # [batch, seq_len, vocab_size]

        return logits


class TransformerBlock(nn.Module):
    """
    Single transformer block using PyTorch's MultiheadAttention.

    Note: nn.TransformerDecoderLayer doesn't work with functional_call,
    so we implement a simple block ourselves.
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.feed_forward = FeedForward(dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        # Create causal mask for attention
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(
            x.device
        )

        # Attention with residual
        normed = self.attention_norm(x)
        attn_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        h = x + attn_out

        # FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class FeedForward(nn.Module):
    """Simple feed-forward network with GELU activation."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ============================================================================
# Model Factory Functions
# ============================================================================


def create_custom_llama(vocab_size=1000, dim=128, n_layers=2, n_heads=4, batch_size=4, seq_len=16):
    """Create custom TinyLLaMA model with random tokens.

    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        batch_size: Batch size for generated tokens
        seq_len: Sequence length for generated tokens

    Returns:
        model: TinyLLaMA model in eval mode
        tokens: Random token tensor [batch_size, seq_len]
    """
    torch.manual_seed(42)

    model = TinyLLaMA(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads)
    model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    return model, tokens


def create_huggingface_model(model_name, max_seq_len=32):
    """Create HuggingFace model with tokenized text.

    Args:
        model_name: HuggingFace model identifier
        max_seq_len: Maximum sequence length for tokenization

    Returns:
        model: HuggingFace model in eval mode
        input_ids: Tokenized input tensor
    """
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers library not installed")

    torch.manual_seed(42)

    try:
        # Load model config and modify dropout settings for deterministic behavior
        config = transformers.AutoConfig.from_pretrained(model_name)

        # Disable dropout for deterministic behavior
        dropout_attrs = [
            "attn_pdrop",
            "resid_pdrop",
            "embd_pdrop",  # GPT-2
            "attention_dropout",
            "hidden_dropout",  # Qwen, LLaMA, Gemma
            "dropout",
            "attn_dropout",
            "ffn_dropout",
        ]
        for attr in dropout_attrs:
            if hasattr(config, attr):
                setattr(config, attr, 0.0)

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception as e:
        pytest.skip(f"Failed to load model '{model_name}': {e}")

    model.eval()

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create sample text data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn patterns from data.",
        "PyTorch is a popular deep learning framework.",
        "Differential privacy protects individual data points.",
    ]

    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    return model, encoding["input_ids"]


# ============================================================================
# Loss Functions
# ============================================================================


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss.

    Args:
        logits: Model logits [batch, seq_len, vocab_size] or model output with .logits attribute
        targets: Target token IDs [batch, seq_len]

    Returns:
        loss: Cross-entropy loss (scalar)
    """
    # Handle HuggingFace model outputs
    if hasattr(logits, "logits"):
        logits = logits.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    # Compute cross-entropy loss
    loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_targets.view(-1))

    return loss
