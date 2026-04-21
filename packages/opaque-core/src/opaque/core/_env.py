"""Internal environment variable helpers."""

from __future__ import annotations

import os


# Allowed tokens for each OPAQUE_SKIP_* env var. Unknown tokens are rejected
# with a ValueError at parse time so typos like `SKIP_PATCHES=vamp` fail loudly
# instead of silently running every patch. Env vars not in this map accept any
# tokens (defensive for callers with custom vocabularies).
ALLOWED_SKIP_TOKENS: dict[str, frozenset[str]] = {
    "OPAQUE_SKIP_PYTORCH_PATCHES": frozenset({"all", "checkpoint"}),
    "OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES": frozenset({"all"}),
    "OPAQUE_SKIP_TRANSFORMERS_PATCHES": frozenset(
        {"all", "vmap", "kv_cache", "batchify", "data"}
    ),
    "OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES": frozenset(
        {"all", "shared", "standard", "gemma2", "phi3"}
    ),
    "OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES": frozenset({"all", "collator"}),
    "OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES": frozenset(
        {"all", "swiglu", "rope", "ce", "fused_ce", "lora"}
    ),
}


def parse_skip_env(name: str) -> set[str]:
    """Parse comma-separated env var values into normalized token set.

    Returns lowercase, whitespace-trimmed, non-empty entries. Raises
    ``ValueError`` if the env var is registered in ``ALLOWED_SKIP_TOKENS``
    and contains any token outside that allowlist.
    """
    raw = os.environ.get(name, "")
    tokens = {entry.strip().lower() for entry in raw.split(",") if entry.strip()}
    allowed = ALLOWED_SKIP_TOKENS.get(name)
    if allowed is not None and (unknown := tokens - allowed):
        raise ValueError(
            f"{name}: unknown token(s) {sorted(unknown)}. "
            f"Expected comma-separated subset of {sorted(allowed)}."
        )
    return tokens
