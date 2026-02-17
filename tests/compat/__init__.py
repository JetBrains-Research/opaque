# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for HuggingFace Transformers with Opaque.

Test Structure:
- conftest.py: Shared fixtures and helpers
- test_attention.py: Attention implementations (eager, sdpa, flash, flex)
- test_features.py: Training features (checkpointing, mixed precision, compile)
- test_peft.py: PEFT methods (LoRA, IA3, Prefix, P-Tuning, Prompt)
- test_architectures.py: Model architectures (Qwen2, Gemma2, DeepSeek, Phi2)

Install: uv sync --group test
Run: pytest tests/compat/ -v

Platform Support:
- macOS: Runs CPU tests (eager, sdpa on CPU)
- Linux with CUDA: Runs all tests including CUDA variants
- CI without GPU: Skips CUDA-only tests automatically
"""
