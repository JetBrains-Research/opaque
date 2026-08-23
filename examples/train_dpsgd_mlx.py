"""Train an MLX-LM causal language model with Opaque DP-SGD.

Model, tokenizer, LoRA, arrays, and runtime integration are native to MLX and
MLX-LM.

Install the example dependencies before running it::

    uv sync --group examples --all-packages --extra all
    uv run python examples/train_dpsgd_mlx.py --preset smoke
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import opaque.accounting as acc
import opaque.auditing as auditing
import opaque.distributed as distributed
import opaque.dpsgd.accounting as dpsgd_acc
import opaque.ops as ops
from opaque.backend import set_backend
from opaque.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
    per_group,
)
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)
from opaque.functional import empty_collate
from opaque.mlx.functional import make_functional
from opaque.optimizers import (
    adafactor,
    adagrad,
    adam,
    adamw,
    ademamix,
    apply_updates,
    lion,
    rmsprop,
    sgd,
)
from opaque.pytree import merge, tree_leaves
from opaque.random import fold_in, key, split
from opaque.scheduling import (
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "model_name": "HuggingFaceTB/SmolLM2-135M",
        "dataset": "JetBrains/KExercises",
        "dataset_text_field": "solution",
        "num_train_samples": 256,
        "num_eval_samples": 64,
        "num_epochs": 1,
        "batch_size": 16,
        "log_steps": 5,
        "eval_steps": 5,
        "target_epsilon": 3.0,
        "learning_rate": 1e-5,
        "lora_r": 4,
        "lora_alpha": 8.0,
        "lora_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_len": 512,
        "dtype": "bfloat16",
        "audit": False,
    },
    "mellum-kstack": {
        "model_name": "JetBrains/Mellum-4b-base",
        "dataset": "JetBrains/KStack",
        "dataset_text_field": "content",
        "num_train_samples": 50_000,
        "num_eval_samples": 1_000,
        "num_epochs": 3,
        "batch_size": 128,
        "log_steps": 2,
        "eval_steps": 10,
        "target_epsilon": 10.0,
        "learning_rate": 5e-5,
        "lora_r": 16,
        "lora_alpha": 32.0,
        "lora_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_seq_len": 1_024,
        "dtype": "bfloat16",
        "microbatch_size": 16,
    },
    "mellum2-kstack": {
        "model_name": "JetBrains/Mellum2-12B-A2.5B-Base",
        "dataset": "JetBrains/KStack",
        "dataset_text_field": "content",
        "num_train_samples": 50_000,
        "num_eval_samples": 1_000,
        "num_epochs": 3,
        "batch_size": 128,
        "log_steps": 2,
        "eval_steps": 10,
        "target_epsilon": 10.0,
        "learning_rate": 5e-5,
        "lora_r": 16,
        "lora_alpha": 32.0,
        "lora_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_len": 1_024,
        "dtype": "bfloat16",
        "microbatch_size": 16,
    },
    "qwen-7b-kstack": {
        "model_name": "Qwen/Qwen2.5-Coder-7B",
        "dataset": "JetBrains/KStack",
        "dataset_text_field": "content",
        "num_train_samples": 50_000,
        "num_eval_samples": 1_000,
        "num_epochs": 2,
        "batch_size": 192,
        "microbatch_size": 16,
        "log_steps": 2,
        "eval_steps": 10,
        "target_epsilon": 3.0,
        "learning_rate": 5e-4,
        "lora_r": 16,
        "lora_alpha": 16.0,
        "lora_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_seq_len": 1_024,
        "dtype": "bfloat16",
    },
}

_DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the native MLX-LM training CLI."""
    parser = argparse.ArgumentParser(
        description="Train a native MLX-LM causal LM with Opaque DP-SGD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(_PRESETS), default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--model-name", default=None)
    model.add_argument("--model-revision", default=None)
    model.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom tokenizer code and MLX-LM model_file code.",
    )
    model.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default=None)
    data.add_argument(
        "--dataset-subset", "--dataset-name", dest="dataset_subset", default=None
    )
    data.add_argument("--dataset-split", default="train")
    data.add_argument("--dataset-text-field", default="text")
    data.add_argument(
        "--dataset-prompt-field",
        default=None,
        help="Optional field whose token prefix is excluded from the loss.",
    )
    data.add_argument("--num-train-samples", type=int, default=5_000)
    data.add_argument(
        "--num-eval-samples",
        "--num-eval-samples-alt",
        dest="num_eval_samples",
        type=int,
        default=1_000,
    )
    data.add_argument("--max-seq-len", type=int, default=512)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--eval-batch-size", type=int, default=None)
    training.add_argument("--num-epochs", type=int, default=3)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument(
        "--lr-schedule", choices=["none", "cosine", "linear", "sqrt"], default="none"
    )
    training.add_argument("--lr-min-ratio", type=float, default=0.0)
    training.add_argument("--lr-warmup-steps", type=int, default=0)
    training.add_argument(
        "--optimizer",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "ademamix",
            "lion",
            "adafactor",
            "rmsprop",
            "adagrad",
        ],
        default="adafactor",
    )
    training.add_argument(
        "--noise-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--log-steps", type=int, default=1)
    training.add_argument("--eval-steps", type=int, default=10)
    training.add_argument("--stop-at-step", type=int, default=None)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--mlx-distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize and use an MLX distributed process group.",
    )
    training.add_argument(
        "--mlx-distributed-backend",
        choices=["any", "ring", "mpi"],
        default="any",
    )

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--lora-r", type=int, default=4)
    lora.add_argument("--lora-alpha", type=float, default=8.0)
    lora.add_argument("--lora-dropout", type=float, default=0.0)
    lora.add_argument("--num-lora-layers", type=int, default=-1)
    lora.add_argument(
        "--lora-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dp = parser.add_argument_group("differential privacy")
    dp.add_argument("--shard", action=argparse.BooleanOptionalAction, default=True)
    dp.add_argument(
        "--clipping-mode", choices=["fixed", "adaptive", "auto"], default="adaptive"
    )
    dp.add_argument("--clipping-norm", type=float, default=1.0)
    dp.add_argument("--target-clipping-rate", type=float, default=0.5)
    dp.add_argument("--clipping-norm-max", type=float, default=10.0)
    dp.add_argument("--auto-clipping-gamma", type=float, default=0.01)
    dp.add_argument("--microbatch-size", type=int, default=None)
    dp.add_argument("--truncated-batch-size", type=int, default=None)
    dp.add_argument(
        "--sampler",
        choices=["poisson", "random_allocation", "k_out_of_t"],
        default="poisson",
    )
    dp.add_argument("--total-participations", dest="K", type=int, default=None)
    dp.add_argument(
        "--noise-mechanism",
        choices=["gaussian", "bounded_gaussian"],
        default="gaussian",
    )
    dp.add_argument("--noise-bound", type=float, default=1.0)
    dp.add_argument(
        "--second-moment", action=argparse.BooleanOptionalAction, default=False
    )
    dp.add_argument("--per-group-clipping", nargs="+", default=None)

    privacy = parser.add_argument_group("privacy")
    privacy.add_argument("--target-epsilon", type=float, default=8.0)
    privacy.add_argument("--target-delta", type=float, default=None)
    privacy.add_argument("--noise-multiplier", type=float, default=None)
    privacy.add_argument("--calibration-min", type=float, default=0.11)
    privacy.add_argument("--calibration-max", type=float, default=3.5)
    privacy.add_argument("--calibration-tolerance", type=float, default=1e-3)

    audit = parser.add_argument_group("privacy auditing")
    audit.add_argument("--audit", action=argparse.BooleanOptionalAction, default=False)
    audit.add_argument("--audit-canaries", type=int, default=1_000)
    audit.add_argument("--audit-method", choices=["gdp", "eps_delta"], default="gdp")
    audit.add_argument("--audit-batch-size", type=int, default=None)

    tracking = parser.add_argument_group("tracking and output")
    tracking.add_argument("--no-wandb", action="store_true", default=False)
    tracking.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "opaque")
    )
    tracking.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
    )
    tracking.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", "federated-compute")
    )
    tracking.add_argument("--checkpoint-path", type=Path, default=None)
    tracking.add_argument("--resume-from", type=Path, default=None)
    tracking.add_argument("--adapter-path", type=Path, default=Path("adapters"))
    return parser


def _provided_destinations(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> set[str]:
    option_destinations = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_destinations:
            provided.add(option_destinations[option])
    return provided


def _parse_per_group_clipping(
    values: Sequence[str] | None,
) -> tuple[dict[str, float] | None, float | None]:
    if values is None:
        return None, None
    groups: dict[str, float] = {}
    fallback = None
    for value in values:
        try:
            pattern, raw_norm = value.split("=", 1)
            norm = float(raw_norm)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid per-group clipping value {value!r}; expected PATTERN=NORM"
            ) from error
        if norm <= 0:
            raise argparse.ArgumentTypeError(
                "per-group clipping norms must be positive"
            )
        if pattern == "fallback":
            fallback = norm
        else:
            groups[pattern] = norm
    return groups, fallback


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments with explicit CLI values taking precedence over presets."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    provided = _provided_destinations(parser, raw_argv)
    args = parser.parse_args(raw_argv)

    if args.preset is not None:
        for name, value in _PRESETS[args.preset].items():
            if name not in provided:
                setattr(args, name, value)

    missing = [
        name for name in ("model_name", "dataset") if getattr(args, name) is None
    ]
    if missing:
        flags = " and ".join(f"--{name.replace('_', '-')}" for name in missing)
        parser.error(
            f"missing required configuration: {flags}. Pass them directly or select "
            "a --preset (for example, --preset smoke)."
        )
    if args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2 for shifted causal-LM loss")
    if args.lora_r <= 0:
        parser.error("--lora-r must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.num_lora_layers == 0 or args.num_lora_layers < -1:
        parser.error("--num-lora-layers must be -1 or a positive integer")
    if args.microbatch_size == 0:
        args.microbatch_size = None
    try:
        args.per_group_clipping, args.per_group_clipping_fallback = (
            _parse_per_group_clipping(args.per_group_clipping)
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    args.eval_batch_size = (
        args.eval_batch_size or args.microbatch_size or args.batch_size
    )
    args.audit_batch_size = (
        args.audit_batch_size or args.microbatch_size or args.batch_size
    )
    return args


def _model_config_path(model_name: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path / "config.json" if local_path.is_dir() else local_path
    return Path(
        hf_hub_download(repo_id=model_name, filename="config.json", revision=revision)
    )


def _preflight_remote_model_code(args: argparse.Namespace) -> None:
    config_path = _model_config_path(args.model_name, args.model_revision)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_file") and not args.trust_remote_code:
        raise ValueError(
            f"{args.model_name!r} declares custom MLX model code in "
            f"{config['model_file']!r}; pass --trust-remote-code to allow execution"
        )


def _ensure_padding_token(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise ValueError(
                "the tokenizer defines neither a pad token nor an EOS token"
            )
        tokenizer.pad_token = eos_token
        pad_token_id = eos_token_id
    return int(pad_token_id)


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a native MLX-LM model and tokenizer after trust preflight."""
    import mlx_lm

    _preflight_remote_model_code(args)
    loaded = mlx_lm.load(
        args.model_name,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        lazy=False,
        return_config=True,
        revision=args.model_revision,
    )
    model, tokenizer, config = loaded
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"mlx_lm.load returned {type(model).__name__}, not mlx.nn.Module"
        )
    _ensure_padding_token(tokenizer)
    model.set_dtype(_DTYPES[args.dtype])
    mx.eval(model.parameters())
    return model, tokenizer, config


def _eligible_lora_module(module: nn.Module) -> bool:
    eligible_types = (
        nn.Linear,
        nn.QuantizedLinear,
        nn.Embedding,
        nn.QuantizedEmbedding,
    )
    return hasattr(module, "to_lora") or isinstance(module, eligible_types)


def _select_lora_keys(
    model: nn.Module, target_patterns: Sequence[str]
) -> tuple[list[str], dict[str, list[str]]]:
    if not hasattr(model, "layers"):
        raise ValueError(
            f"{type(model).__name__} has no layers sequence required by MLX-LM LoRA"
        )
    available = {
        name
        for layer in model.layers
        for name, module in layer.named_modules()
        if name and _eligible_lora_module(module)
    }
    matches = {
        pattern: sorted(name for name in available if pattern in name)
        for pattern in target_patterns
    }
    missing = [pattern for pattern, names in matches.items() if not names]
    if missing:
        preview = ", ".join(sorted(available)[:20]) or "<none>"
        raise ValueError(
            f"LoRA target pattern(s) {missing!r} matched no eligible modules; "
            f"available module keys include: {preview}"
        )
    selected = sorted({name for names in matches.values() for name in names})
    return selected, matches


def prepare_lora(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    """Freeze a model and replace selected native MLX-LM layers with LoRA."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if not hasattr(model, "layers"):
        raise ValueError(
            f"{type(model).__name__} has no layers sequence required by MLX-LM LoRA"
        )
    if args.num_lora_layers > len(model.layers):
        raise ValueError(
            f"requested {args.num_lora_layers} LoRA layers, but the model has "
            f"{len(model.layers)} layers"
        )
    model.freeze()
    keys, matches = _select_lora_keys(model, args.lora_modules)
    lora_parameters = {
        "rank": args.lora_r,
        "scale": args.lora_alpha / args.lora_r,
        "dropout": args.lora_dropout,
        "keys": keys,
    }
    linear_to_lora_layers(
        model,
        num_layers=args.num_lora_layers,
        config=lora_parameters,
    )
    trainable = model.trainable_parameters()
    if not trainable:
        raise ValueError("MLX-LM LoRA conversion produced no trainable parameters")
    return {
        "fine_tune_type": "lora",
        "num_layers": args.num_lora_layers,
        "lora_parameters": lora_parameters,
        "target_matches": matches,
    }


def functionalize_model(
    model: nn.Module,
) -> tuple[Callable[..., mx.array], Mapping[str, Any], Mapping[str, Any]]:
    """Expose sparse LoRA parameters while retaining the frozen base tree."""
    return make_functional(model, partition_trainable=True)


def _load_streaming_subset(args: argparse.Namespace) -> Any:
    from datasets import Dataset, load_dataset

    total_needed = args.num_train_samples + args.num_eval_samples
    stream = load_dataset(
        args.dataset,
        name=args.dataset_subset,
        split=args.dataset_split,
        streaming=True,
    )
    rows = list(stream.take(total_needed))
    required_fields = [args.dataset_text_field]
    if args.dataset_prompt_field is not None:
        required_fields.append(args.dataset_prompt_field)
    if rows:
        missing = [field for field in required_fields if field not in rows[0]]
        if missing:
            raise ValueError(
                f"dataset row is missing field(s) {missing!r}; available fields: "
                f"{sorted(rows[0])}"
            )
    if len(rows) < total_needed:
        raise ValueError(
            f"dataset supplied {len(rows)} rows, but {total_needed} are required "
            "for the requested train and evaluation subsets"
        )
    return Dataset.from_list(rows)


def tokenize_record(
    record: Mapping[str, Any], tokenizer: Any, args: argparse.Namespace
) -> dict[str, list[int] | list[bool]]:
    """Tokenize one privacy unit into a deterministic fixed-length record."""
    tokenizer_call = (
        tokenizer if callable(tokenizer) else getattr(tokenizer, "_tokenizer", None)
    )
    if tokenizer_call is None or not callable(tokenizer_call):
        raise TypeError("tokenizer must be callable or wrap a callable HF tokenizer")
    text = record[args.dataset_text_field]
    if not isinstance(text, str):
        raise TypeError(
            f"dataset field {args.dataset_text_field!r} must contain strings"
        )
    encoded = tokenizer_call(
        text,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
        return_attention_mask=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = [bool(value) for value in encoded["attention_mask"]]
    if len(input_ids) != args.max_seq_len or len(attention_mask) != args.max_seq_len:
        raise ValueError("tokenizer did not honor fixed max-length padding")

    loss_mask = attention_mask
    if args.dataset_prompt_field is not None:
        prompt = record[args.dataset_prompt_field]
        if not isinstance(prompt, str):
            raise TypeError(
                f"dataset field {args.dataset_prompt_field!r} must contain strings"
            )
        prompt_ids = tokenizer_call(
            prompt,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=True,
        )["input_ids"]
        for index in range(min(len(prompt_ids), len(loss_mask))):
            loss_mask[index] = False

    if not any(loss_mask[1:]):
        raise ValueError(
            "record has no trainable target token after shifting and masking"
        )
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def prepare_datasets(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[Any, Any, Callable[[list[Mapping[str, Any]]], dict[str, mx.array]]]:
    """Materialize, split, tokenize, and warm a fixed-shape MLX collator."""
    dataset = _load_streaming_subset(args)
    eval_dataset = dataset.take(args.num_eval_samples)
    train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)
    remove_columns = dataset.column_names

    def tokenize(row: Mapping[str, Any]) -> dict[str, list[int] | list[bool]]:
        return tokenize_record(row, tokenizer, args)

    eval_dataset = eval_dataset.map(tokenize, remove_columns=remove_columns)
    train_dataset = train_dataset.map(tokenize, remove_columns=remove_columns)
    collate = empty_collate(collate_records)
    representative = train_dataset[0] if len(train_dataset) else eval_dataset[0]
    collate([representative])
    return train_dataset, eval_dataset, collate


def collate_records(records: list[Mapping[str, Any]]) -> dict[str, mx.array]:
    """Stack already fixed-shape records without changing the privacy unit."""
    if not records:
        raise ValueError("collate_records must be warmed with a non-empty batch")
    return {
        "input_ids": mx.array(
            [record["input_ids"] for record in records], dtype=mx.int32
        ),
        "loss_mask": mx.array(
            [record["loss_mask"] for record in records], dtype=mx.bool_
        ),
    }


def build_record_loss(
    functional_model: Callable[..., mx.array],
) -> Callable[[Mapping[str, Any], mx.array, mx.array], mx.array]:
    """Build scalar shifted causal-LM loss for one complete dataset record."""

    def record_loss(
        params: Mapping[str, Any], input_ids: mx.array, loss_mask: mx.array
    ) -> mx.array:
        logits = functional_model(params, input_ids[None, :-1])
        targets = input_ids[None, 1:]
        shifted_mask = loss_mask[None, 1:]
        token_losses = nn.losses.cross_entropy(logits, targets)
        token_count = mx.sum(shifted_mask)
        return mx.sum(token_losses.astype(mx.float32) * shifted_mask) / token_count

    return record_loss


def bind_explicit_parameters(
    model: nn.Module,
    trainable: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    """Make the final explicit tree authoritative for evaluation and export."""
    model.update(merge(frozen, trainable))
    mx.eval(model.parameters())


def evaluate_model(
    functional_model: Callable[..., mx.array],
    params: Mapping[str, Any],
    dataset: Any,
    collate: Callable[[list[Mapping[str, Any]]], Mapping[str, mx.array]],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate token-normalized shifted CE and perplexity."""
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(dataset), batch_size):
        records = [
            dataset[index]
            for index in range(start, min(len(dataset), start + batch_size))
        ]
        batch = collate(records)
        logits = functional_model(params, batch["input_ids"][:, :-1])
        targets = batch["input_ids"][:, 1:]
        mask = batch["loss_mask"][:, 1:]
        losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
        loss_sum = mx.sum(losses * mask)
        token_count = mx.sum(mask)
        mx.eval(loss_sum, token_count)
        count = int(token_count.item())
        if count:
            total_loss += float(loss_sum.item())
            total_tokens += count
    mean_loss = total_loss / total_tokens if total_tokens else math.nan
    perplexity = math.exp(mean_loss) if math.isfinite(mean_loss) else math.nan
    return {"loss": mean_loss, "perplexity": perplexity, "tokens": float(total_tokens)}


@dataclass(frozen=True)
class AuditContext:
    """Canary partition and reference scores retained across training."""

    dataset: Any
    coin_flip: Any
    reference_scores: Any | None


def prepare_audit(
    args: argparse.Namespace,
    *,
    loss_fn: Callable[..., mx.array],
    params: Mapping[str, Any],
    dataset: Any,
    collate: Callable[[list[Mapping[str, Any]]], Mapping[str, mx.array]],
) -> tuple[Any, AuditContext | None]:
    """Hold out out-canaries and capture untrained reference scores."""
    if not args.audit:
        return dataset, None
    if args.audit_canaries <= 0 or args.audit_canaries > len(dataset):
        raise ValueError("--audit-canaries must be in [1, number of train records]")
    coin_flip = auditing.coin_flip(
        dataset, num_canaries=args.audit_canaries, key=key(args.seed)
    )
    train_indices = coin_flip.train_indices(len(dataset))
    training_dataset = (
        dataset.select(train_indices)
        if hasattr(dataset, "select")
        else [dataset[int(index)] for index in train_indices]
    )

    def audit_collate(records: list[Mapping[str, Any]]) -> tuple[mx.array, mx.array]:
        batch = collate(records)
        return batch["input_ids"], batch["loss_mask"]

    reference_scores = None
    if distributed.is_main_process():
        reference_scores = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=coin_flip,
            dataset=dataset,
            batch_size=args.audit_batch_size,
            collate_fn=audit_collate,
        )
    return training_dataset, AuditContext(dataset, coin_flip, reference_scores)


def evaluate_audit(
    args: argparse.Namespace,
    context: AuditContext,
    *,
    loss_fn: Callable[..., mx.array],
    params: Mapping[str, Any],
    collate: Callable[[list[Mapping[str, Any]]], Mapping[str, mx.array]],
    delta: float,
) -> dict[str, float]:
    """Compute loss-reduction membership scores and a one-run estimate."""

    def audit_collate(records: list[Mapping[str, Any]]) -> tuple[mx.array, mx.array]:
        batch = collate(records)
        return batch["input_ids"], batch["loss_mask"]

    scores = auditing.loss_scores(
        loss_fn,
        params,
        batch_argnums=(1, 2),
        coin_flip=context.coin_flip,
        dataset=context.dataset,
        reference_scores=context.reference_scores,
        batch_size=args.audit_batch_size,
        collate_fn=audit_collate,
    )
    estimate = auditing.one_run(scores, coin_flip=context.coin_flip)
    method = estimate.gdp() if args.audit_method == "gdp" else estimate.eps_delta()
    return {
        "epsilon": float(method.epsilon_at(delta=delta)),
        "auc": float(estimate.attack_auc()),
        "beta_0.01": float(estimate.attack_beta_at(alpha=0.01)),
        "beta_0.10": float(estimate.attack_beta_at(alpha=0.10)),
        "n_in": float(estimate.n_in),
        "n_out": float(estimate.n_out),
    }


def export_mlx_lm_adapter(
    path: Path,
    model: nn.Module,
    adapter_config: Mapping[str, Any],
) -> None:
    """Write the standard MLX-LM adapter config and trainable safetensors."""
    from mlx.utils import tree_flatten

    path.mkdir(parents=True, exist_ok=True)
    export_config = {
        "fine_tune_type": adapter_config["fine_tune_type"],
        "num_layers": adapter_config["num_layers"],
        "lora_parameters": adapter_config["lora_parameters"],
    }
    (path / "adapter_config.json").write_text(
        json.dumps(export_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    weights = dict(tree_flatten(model.trainable_parameters()))
    if not weights:
        raise ValueError("cannot export an adapter without trainable parameters")
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)


def verify_adapter_reload(
    args: argparse.Namespace,
    adapter_path: Path,
    tokenizer: Any,
) -> None:
    """Reload the exported adapter through MLX-LM and execute one forward pass."""
    import mlx_lm

    model, _ = mlx_lm.load(
        args.model_name,
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        lazy=False,
        revision=args.model_revision,
    )
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        token_id = _ensure_padding_token(tokenizer)
    logits = model(mx.array([[int(token_id)]], dtype=mx.int32))
    mx.eval(logits)
    if logits.shape[:2] != (1, 1):
        raise RuntimeError(
            f"reloaded adapter returned unexpected logits shape {logits.shape}"
        )


@dataclass(frozen=True)
class TrainingPlan:
    """Static horizon and privacy parameters shared by all training state."""

    dataset_size: int
    sample_rate: float
    steps_per_epoch: int
    total_steps: int
    num_bins: int
    target_delta: float
    noise_multiplier: float


@dataclass(frozen=True)
class PrivateMechanisms:
    """Opaque transforms whose mutable values live in explicit state."""

    grad_fn: Callable[..., Any]
    noise_fn: Callable[..., Any]
    optimizer_step: Callable[..., Any]
    accounting_step: Callable[[float], Any]
    clip_norm: float | PerGroup
    learning_rate: float | Callable[[int], float]


@dataclass(frozen=True)
class PrivateTrainingState:
    """Complete explicit state needed to continue private optimization."""

    params: Mapping[str, Any]
    clip_state: Any
    noise_state: Any
    optimizer_state: Any
    accountant: acc.Accountant
    step: int = 0


_SECOND_MOMENT_OPTIMIZERS = {"adam", "adamw", "ademamix", "rmsprop"}


def validate_private_configuration(args: argparse.Namespace) -> None:
    """Reject mechanism combinations whose runtime and accountant disagree."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive")
    if args.stop_at_step is not None and args.stop_at_step <= 0:
        raise ValueError("--stop-at-step must be positive")
    if args.log_steps <= 0 or args.eval_steps <= 0:
        raise ValueError("--log-steps and --eval-steps must be positive")
    if args.eval_batch_size <= 0 or args.audit_batch_size <= 0:
        raise ValueError("evaluation and audit batch sizes must be positive")
    if args.microbatch_size is not None and args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive or zero for disabled")
    if args.truncated_batch_size is not None:
        if args.truncated_batch_size <= 0:
            raise ValueError("--truncated-batch-size must be positive")
        if args.sampler != "poisson":
            raise ValueError("truncated Poisson batches require --sampler poisson")
    if args.sampler == "k_out_of_t" and args.K is None:
        raise ValueError("--total-participations is required for k_out_of_t")
    if args.sampler != "k_out_of_t" and args.K is not None:
        raise ValueError("--total-participations is only valid for k_out_of_t")
    if not 0.0 <= args.lr_min_ratio <= 1.0:
        raise ValueError("--lr-min-ratio must be in [0, 1]")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be non-negative")
    if args.second_moment and args.optimizer not in _SECOND_MOMENT_OPTIMIZERS:
        compatible = ", ".join(sorted(_SECOND_MOMENT_OPTIMIZERS))
        raise ValueError(
            f"private second moments require a compatible optimizer: {compatible}"
        )


def validate_distributed_configuration(args: argparse.Namespace) -> None:
    """Reject distributed modes whose privacy equivalence is not implemented."""
    if not distributed.is_distributed():
        return
    if not args.shard:
        raise ValueError(
            "distributed MLX training requires --shard; parallel Poisson "
            "without record sharding is not implemented"
        )
    if args.sampler != "poisson":
        raise ValueError(
            "distributed MLX training currently supports only Poisson sampling"
        )


def build_training_plan(
    args: argparse.Namespace,
    *,
    dataset_size: int,
    accounting_step: Callable[[float], Any],
) -> TrainingPlan:
    """Resolve a fixed training horizon and calibrate its noise multiplier."""
    validate_private_configuration(args)
    if dataset_size <= 0:
        raise ValueError("the private training dataset must not be empty")
    if args.batch_size >= dataset_size:
        raise ValueError(
            "--batch-size must be smaller than the private dataset size so the "
            "sampling rate is in (0, 1)"
        )
    steps_per_epoch = math.ceil(dataset_size / args.batch_size)
    total_steps = args.num_epochs * steps_per_epoch
    if args.sampler == "k_out_of_t" and not 1 <= args.K <= total_steps:
        raise ValueError(
            f"--total-participations must be in [1, {total_steps}], got {args.K}"
        )
    target_delta = args.target_delta or 1.0 / (dataset_size**1.1)
    if not 0.0 < target_delta < 1.0:
        raise ValueError("--target-delta must be in (0, 1)")

    if args.noise_multiplier is None:
        result = acc.calibrate(
            acc.epsilon_budget(args.target_epsilon, delta=target_delta),
            lambda multiplier: accounting_step(multiplier) * total_steps,
            param_min=args.calibration_min,
            param_max=args.calibration_max,
            tolerance=args.calibration_tolerance,
        )
        if not result.converged:
            raise RuntimeError(
                "privacy calibration did not converge; widen "
                "--calibration-min/--calibration-max"
            )
        noise_multiplier = result.param
    else:
        if args.noise_multiplier < 0:
            raise ValueError("--noise-multiplier must be non-negative")
        noise_multiplier = args.noise_multiplier

    return TrainingPlan(
        dataset_size=dataset_size,
        sample_rate=args.batch_size / dataset_size,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        num_bins=max(2, steps_per_epoch),
        target_delta=target_delta,
        noise_multiplier=noise_multiplier,
    )


def resolve_clip_norm(
    args: argparse.Namespace, params: Mapping[str, Any]
) -> float | PerGroup:
    """Resolve scalar or path-pattern clipping bounds against trainable leaves."""
    if args.per_group_clipping is None:
        return args.clipping_norm
    return per_group(
        params,
        args.per_group_clipping,
        fallback=args.per_group_clipping_fallback,
    )


def build_accounting_step(
    args: argparse.Namespace,
    *,
    dataset_size: int,
    total_steps: int,
    num_bins: int,
    clip_norm: float | PerGroup,
) -> Callable[[float], Any]:
    """Pair the selected sampler and clipping release with its accountant."""
    sample_rate = args.batch_size / dataset_size

    def unamplified(noise_multiplier: float) -> Any:
        base = (
            acc.nonprivate()
            if noise_multiplier == 0
            else dpsgd_acc.gaussian(noise_multiplier)
        )
        if args.clipping_mode == "adaptive":
            num_groups = len(clip_norm.values) if isinstance(clip_norm, PerGroup) else 1
            return dpsgd_acc.adaclip(
                base,
                expected_batch_size=args.batch_size,
                num_groups=num_groups,
            )
        return base

    if args.sampler == "poisson":
        return lambda multiplier: dpsgd_acc.poisson(
            unamplified(multiplier),
            sample_rate=sample_rate,
            truncated_batch_size=args.truncated_batch_size,
            dataset_size=(
                dataset_size if args.truncated_batch_size is not None else None
            ),
        )
    if args.sampler == "random_allocation":
        return lambda multiplier: acc.per_step(
            dpsgd_acc.random_allocation(
                unamplified(multiplier),
                num_bins=num_bins,
                n_steps=total_steps,
            )
        )
    return lambda multiplier: acc.per_step(
        dpsgd_acc.k_out_of_t(
            unamplified(multiplier),
            total_participations=args.K,
            n_steps=total_steps,
        )
    )


def build_learning_rate(
    args: argparse.Namespace, total_steps: int
) -> float | Callable[[int], float]:
    """Build the fixed-horizon learning-rate schedule."""
    peak = args.learning_rate
    minimum = peak * args.lr_min_ratio
    warmup = args.lr_warmup_steps
    decay_steps = max(1, total_steps - warmup)
    if args.lr_schedule == "none":
        schedule: float | Callable[[int], float] = peak
    elif args.lr_schedule == "cosine":
        schedule = cosine_schedule(
            peak,
            minimum,
            transition_steps=decay_steps,
            transition_begin=warmup,
        )
    elif args.lr_schedule == "linear":
        schedule = linear_schedule(
            peak,
            minimum,
            transition_steps=decay_steps,
            transition_begin=warmup,
        )
    else:
        schedule = inverse_sqrt_schedule(
            peak,
            transition_steps=warmup if warmup > 0 else max(1, total_steps),
            transition_begin=warmup,
        )
    if warmup > 0:
        schedule = with_warmup(schedule, transition_steps=warmup)
    return schedule


def build_optimizer(
    args: argparse.Namespace,
    params: Mapping[str, Any],
    learning_rate: float | Callable[[int], float],
) -> tuple[Callable[..., Any], Any]:
    """Construct an Opaque optimizer that consumes noise metadata."""
    factories = {
        "sgd": sgd,
        "adam": adam,
        "adamw": adamw,
        "ademamix": ademamix,
        "lion": lion,
        "adafactor": adafactor,
        "rmsprop": rmsprop,
        "adagrad": adagrad,
    }
    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "weight_decay": args.weight_decay,
    }
    if args.optimizer not in {"sgd", "lion"}:
        kwargs["noise_bias_correction"] = args.noise_bias_correction
    return factories[args.optimizer](params, **kwargs)


def build_private_mechanisms(
    args: argparse.Namespace,
    *,
    params: Mapping[str, Any],
    loss_fn: Callable[..., mx.array],
    dataset_size: int,
) -> tuple[PrivateMechanisms, PrivateTrainingState, TrainingPlan]:
    """Construct matched clipping, noise, optimizer, sampling, and accounting."""
    validate_private_configuration(args)
    clip_norm = resolve_clip_norm(args, params)
    steps_per_epoch = math.ceil(dataset_size / args.batch_size)
    total_steps = args.num_epochs * steps_per_epoch
    num_bins = max(2, steps_per_epoch)
    accounting_step = build_accounting_step(
        args,
        dataset_size=dataset_size,
        total_steps=total_steps,
        num_bins=num_bins,
        clip_norm=clip_norm,
    )
    plan = build_training_plan(
        args,
        dataset_size=dataset_size,
        accounting_step=accounting_step,
    )

    quantile_key = gradient_noise_key = key(args.seed)
    if args.clipping_mode == "adaptive":
        quantile_key, gradient_noise_key = split(gradient_noise_key)
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            initial_clipping_norm=clip_norm,
            target_quantile=args.target_clipping_rate,
            clipping_norm_max=args.clipping_norm_max,
            key=quantile_key,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=args.second_moment,
        )
    elif args.clipping_mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=clip_norm,
            gamma=args.auto_clipping_gamma,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=args.second_moment,
        )
    else:
        grad_fn, clip_state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=args.second_moment,
        )
    noise_fn, noise_state = gaussian_noise(
        noise_multiplier=plan.noise_multiplier,
        key=gradient_noise_key,
        bound=args.noise_bound if args.noise_mechanism == "bounded_gaussian" else None,
    )
    learning_rate = build_learning_rate(args, plan.total_steps)
    optimizer_step, optimizer_state = build_optimizer(args, params, learning_rate)
    mechanisms = PrivateMechanisms(
        grad_fn=grad_fn,
        noise_fn=noise_fn,
        optimizer_step=optimizer_step,
        accounting_step=accounting_step,
        clip_norm=clip_norm,
        learning_rate=learning_rate,
    )
    state = PrivateTrainingState(
        params=params,
        clip_state=clip_state,
        noise_state=noise_state,
        optimizer_state=optimizer_state,
        accountant=acc.Accountant(),
    )
    return mechanisms, state, plan


def build_sampler(
    args: argparse.Namespace,
    dataset: Any,
    plan: TrainingPlan,
    *,
    epoch: int = 0,
) -> Any:
    """Build the runtime sampler paired with ``build_accounting_step``."""
    if args.sampler == "poisson":
        return PoissonSampler(
            dataset,
            sample_rate=plan.sample_rate,
            truncated_batch_size=args.truncated_batch_size,
            n_steps=plan.steps_per_epoch,
            key=fold_in(key(args.seed), distributed.get_rank(), epoch),
        )
    if args.sampler == "random_allocation":
        return RandomAllocationSampler(
            dataset,
            num_bins=plan.num_bins,
            n_steps=plan.total_steps,
            key=key(args.seed),
        )
    return KOutOfTSampler(
        dataset,
        total_participations=args.K,
        n_steps=plan.total_steps,
        key=key(args.seed),
    )


def iter_sampled_indices(
    args: argparse.Namespace, dataset: Any, plan: TrainingPlan
) -> Any:
    """Yield exactly one index list for each accounted training slot."""
    if args.sampler != "poisson":
        yield from build_sampler(args, dataset, plan)
        return
    for epoch in range(args.num_epochs):
        yield from build_sampler(args, dataset, plan, epoch=epoch)


def _array_leaves(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            leaf
            for field in fields(value)
            for leaf in _array_leaves(getattr(value, field.name))
        ]
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _array_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _array_leaves(item)]
    return [leaf for leaf in tree_leaves(value) if isinstance(leaf, mx.array)]


def realize_training_state(state: PrivateTrainingState, *extra: Any) -> None:
    """Materialize every MLX leaf needed by the next private update."""
    leaves = _array_leaves(state.params)
    leaves.extend(_array_leaves(state.clip_state))
    leaves.extend(_array_leaves(state.noise_state))
    leaves.extend(_array_leaves(state.optimizer_state))
    for value in extra:
        leaves.extend(_array_leaves(value))
    if leaves:
        mx.eval(*leaves)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_training_checkpoint(
    path: Path,
    *,
    state: PrivateTrainingState,
    args: argparse.Namespace,
) -> None:
    """Save resumable Opaque state as JSON metadata plus MLX safetensors."""
    if not distributed.is_main_process():
        return
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}

    def split_tree(value: Any, name: str) -> Any:
        if ops.is_array(value) or isinstance(value, np.ndarray):
            tensors[name] = value if isinstance(value, mx.array) else mx.array(value)
            return {"__tensor__": name}
        if isinstance(value, Mapping):
            return {
                str(child_name): split_tree(child, f"{name}.{child_name}")
                for child_name, child in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                split_tree(child, f"{name}.{index}")
                for index, child in enumerate(value)
            ]
        return _json_safe(value)

    components = {
        "params": state.params,
        "optimizer": state.optimizer_state,
        "clipping": state.clip_state,
        "noise": state.noise_state,
        "accountant": state.accountant,
    }
    structural_state = {
        component: split_tree(state_dict(value), component)
        for component, value in components.items()
    }
    if tensors:
        mx.save_safetensors(str(path / "state.safetensors"), tensors)
    metadata = {
        "format_version": 1,
        "step": state.step,
        "sampler_state": {
            "step": state.step,
            "seed": args.seed,
            "sampler": args.sampler,
        },
        "config": _json_safe(vars(args)),
        "state": structural_state,
    }
    (path / "state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_training_checkpoint(
    path: Path,
    *,
    template: PrivateTrainingState,
    args: argparse.Namespace,
) -> PrivateTrainingState:
    """Restore every explicit training component into fresh runtime templates."""
    metadata_path = path / "state.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"checkpoint metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    saved_sampler = metadata["sampler_state"]
    if saved_sampler["sampler"] != args.sampler or saved_sampler["seed"] != args.seed:
        raise ValueError("resume requires the same --sampler and --seed")
    tensors_path = path / "state.safetensors"
    tensors = mx.load(str(tensors_path)) if tensors_path.exists() else {}

    def restore_tree(value: Any) -> Any:
        if isinstance(value, Mapping) and set(value) == {"__tensor__"}:
            return tensors[value["__tensor__"]]
        if isinstance(value, Mapping):
            return {name: restore_tree(item) for name, item in value.items()}
        if isinstance(value, list):
            return [restore_tree(item) for item in value]
        return value

    restored_components = {
        name: restore_tree(value) for name, value in metadata["state"].items()
    }

    def restore_parameter_tree(template_value: Any, prefix: str = "") -> Any:
        flat_parameters = restored_components["params"]
        if isinstance(template_value, Mapping):
            return {
                name: restore_parameter_tree(
                    value, f"{prefix}.{name}" if prefix else str(name)
                )
                for name, value in template_value.items()
            }
        if isinstance(template_value, (tuple, list)):
            restored_values = [
                restore_parameter_tree(value, f"{prefix}[{index}]")
                for index, value in enumerate(template_value)
            ]
            return type(template_value)(restored_values)
        return flat_parameters[prefix]

    restored = PrivateTrainingState(
        params=restore_parameter_tree(template.params),
        optimizer_state=from_state_dict(
            template.optimizer_state, restored_components["optimizer"]
        ),
        clip_state=from_state_dict(
            template.clip_state, restored_components["clipping"]
        ),
        noise_state=from_state_dict(template.noise_state, restored_components["noise"]),
        accountant=from_state_dict(
            template.accountant, restored_components["accountant"]
        ),
        step=int(metadata["step"]),
    )
    realize_training_state(restored)
    return restored


def private_training_step(
    mechanisms: PrivateMechanisms,
    state: PrivateTrainingState,
    batch: Mapping[str, mx.array],
    *,
    noise_multiplier: float,
) -> tuple[PrivateTrainingState, Any]:
    """Execute one accounted private update, including a zero-record draw."""
    accountant = state.accountant | mechanisms.accounting_step(noise_multiplier)
    (clipped_grads, aux), clip_state = mechanisms.grad_fn(
        state.params,
        batch["input_ids"],
        batch["loss_mask"],
        state=state.clip_state,
    )
    if distributed.is_distributed():
        clip_state = distributed.sync(clip_state)
        aux = distributed.sync(aux)
        clipped_grads = distributed.sum_gradients(clipped_grads)
    noisy_grads, noise_state = mechanisms.noise_fn(clipped_grads, state.noise_state)
    if distributed.is_distributed():
        noise_state = distributed.sync(noise_state)
    updates, optimizer_state = mechanisms.optimizer_step(
        noisy_grads,
        state.optimizer_state,
        params=state.params,
    )
    params = apply_updates(state.params, updates)
    next_state = PrivateTrainingState(
        params=params,
        clip_state=clip_state,
        noise_state=noise_state,
        optimizer_state=optimizer_state,
        accountant=accountant,
        step=state.step + 1,
    )
    realize_training_state(next_state, aux)
    return next_state, aux


def run_private_training(
    args: argparse.Namespace,
    *,
    functional_model: Callable[..., mx.array],
    params: Mapping[str, Any],
    train_dataset: Any,
    collate: Callable[[list[Mapping[str, Any]]], Mapping[str, mx.array]],
    global_dataset_size: int | None = None,
    on_step: Callable[[PrivateTrainingState, Any, int, float, TrainingPlan], None]
    | None = None,
) -> tuple[PrivateTrainingState, TrainingPlan]:
    """Run the native MLX private update loop."""
    loss_fn = build_record_loss(functional_model)
    mechanisms, state, plan = build_private_mechanisms(
        args,
        params=params,
        loss_fn=loss_fn,
        dataset_size=global_dataset_size or len(train_dataset),
    )
    if args.resume_from is not None:
        state = load_training_checkpoint(
            args.resume_from,
            template=state,
            args=args,
        )
    if state.step >= plan.total_steps:
        return state, plan
    for slot, indices in enumerate(iter_sampled_indices(args, train_dataset, plan)):
        if slot < state.step:
            continue
        records = [train_dataset[int(index)] for index in indices]
        batch = collate(records)
        started_at = time.perf_counter()
        state, _ = private_training_step(
            mechanisms,
            state,
            batch,
            noise_multiplier=plan.noise_multiplier,
        )
        elapsed = time.perf_counter() - started_at
        if on_step is not None:
            on_step(state, _, len(records), elapsed, plan)
        if args.stop_at_step is not None and state.step >= args.stop_at_step:
            break
    return state, plan


def main(argv: Sequence[str] | None = None) -> None:
    """Train native MLX-LM LoRA parameters with Opaque DP-SGD."""
    args = parse_args(argv)
    set_backend("mlx")
    mx.random.seed(args.seed)
    if args.mlx_distributed:
        from opaque.mlx import distributed as mlx_distributed

        mlx_distributed.initialize(
            strict=True,
            backend=args.mlx_distributed_backend,
        )
    validate_distributed_configuration(args)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_main = distributed.is_main_process()

    model, tokenizer, _ = load_model_and_tokenizer(args)
    adapter_config = prepare_lora(model, args)
    functional_model, trainable, frozen = functionalize_model(model)
    train_dataset, eval_dataset, collate = prepare_datasets(args, tokenizer)
    loss_fn = build_record_loss(functional_model)
    train_dataset, audit_context = prepare_audit(
        args,
        loss_fn=loss_fn,
        params=trainable,
        dataset=train_dataset,
        collate=collate,
    )
    global_dataset_size = len(train_dataset)
    if distributed.is_distributed():
        train_dataset = distributed.local_shard(train_dataset)

    wandb_run = None
    if is_main and not args.no_wandb:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking is enabled but wandb is not installed; pass --no-wandb"
            ) from error
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=_json_safe(vars(args)),
        )

    if is_main:
        print(
            "Prepared MLX-LM DP-SGD inputs: "
            f"train_records={global_dataset_size} eval_records={len(eval_dataset)} "
            f"lora_targets={len(adapter_config['lora_parameters']['keys'])} "
            f"trainable_roots={len(trainable)} frozen_roots={len(frozen)} "
            f"rank={rank}/{world_size}"
        )
        baseline = evaluate_model(
            functional_model,
            trainable,
            eval_dataset,
            collate,
            batch_size=args.eval_batch_size,
        )
        print(
            f"Initial evaluation: loss={baseline['loss']:.6g} "
            f"perplexity={baseline['perplexity']:.6g}"
        )

    def on_step(
        step_state: PrivateTrainingState,
        aux: Any,
        realized_batch_size: int,
        elapsed: float,
        plan: TrainingPlan,
    ) -> None:
        if not is_main:
            return
        should_log = step_state.step % args.log_steps == 0
        should_evaluate = step_state.step % args.eval_steps == 0
        metrics: dict[str, float] = {
            "train/step": float(step_state.step),
            "train/batch_size": float(realized_batch_size),
            "train/step_seconds": elapsed,
            "privacy/epsilon": float(
                step_state.accountant.epsilon_at(plan.target_delta)
            ),
            "privacy/delta": plan.target_delta,
        }
        if realized_batch_size and hasattr(aux, "loss_values"):
            loss = mx.mean(aux.loss_values)
            grad_norm = mx.mean(aux.grad_norms)
            mx.eval(loss, grad_norm)
            metrics["train/loss"] = float(loss.item())
            metrics["train/grad_norm"] = float(grad_norm.item())
        if should_evaluate:
            evaluation = evaluate_model(
                functional_model,
                step_state.params,
                eval_dataset,
                collate,
                batch_size=args.eval_batch_size,
            )
            metrics.update(
                {
                    "eval/loss": evaluation["loss"],
                    "eval/perplexity": evaluation["perplexity"],
                }
            )
            if args.checkpoint_path is not None:
                save_training_checkpoint(
                    args.checkpoint_path,
                    state=step_state,
                    args=args,
                )
        if should_log:
            formatted = " ".join(
                f"{name}={value:.6g}" for name, value in sorted(metrics.items())
            )
            print(formatted)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step_state.step)

    state, plan = run_private_training(
        args,
        functional_model=functional_model,
        params=trainable,
        train_dataset=train_dataset,
        collate=collate,
        global_dataset_size=global_dataset_size,
        on_step=on_step,
    )

    if args.checkpoint_path is not None:
        save_training_checkpoint(args.checkpoint_path, state=state, args=args)
    bind_explicit_parameters(model, state.params, frozen)
    if is_main:
        final_evaluation = evaluate_model(
            functional_model,
            state.params,
            eval_dataset,
            collate,
            batch_size=args.eval_batch_size,
        )
        final_metrics = {
            "eval/final_loss": final_evaluation["loss"],
            "eval/final_perplexity": final_evaluation["perplexity"],
            "privacy/final_epsilon": float(
                state.accountant.epsilon_at(plan.target_delta)
            ),
        }
        if audit_context is not None:
            audit_metrics = evaluate_audit(
                args,
                audit_context,
                loss_fn=loss_fn,
                params=state.params,
                collate=collate,
                delta=plan.target_delta,
            )
            final_metrics.update(
                {f"audit/{name}": value for name, value in audit_metrics.items()}
            )
            print(
                "Audit: "
                f"method={args.audit_method} epsilon={audit_metrics['epsilon']:.6g} "
                f"auc={audit_metrics['auc']:.6g} n_in={audit_metrics['n_in']:.0f} "
                f"n_out={audit_metrics['n_out']:.0f}"
            )
        export_mlx_lm_adapter(args.adapter_path, model, adapter_config)
        verify_adapter_reload(args, args.adapter_path, tokenizer)
        print(
            f"Completed {state.step}/{plan.total_steps} private updates: "
            f"epsilon={final_metrics['privacy/final_epsilon']:.6g} "
            f"delta={plan.target_delta:.3g} "
            f"eval_loss={final_evaluation['loss']:.6g} "
            f"adapter={args.adapter_path}"
        )
        if wandb_run is not None:
            wandb_run.log(final_metrics, step=state.step)
            wandb_run.finish()
    distributed.wait_for_everyone()


if __name__ == "__main__":
    main()
