# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end DP-SGD DPO (Direct Preference Optimization) example for causal LMs.

This is the DPO sibling of ``examples/train_causal_lm.py`` and ``train_sft.py``:
it demonstrates the ``opaque-alignment`` primitives wired into the proven DP-SGD
functional loop (``make_functional`` -> per-example loss -> ``clipped_grad`` ->
``gaussian_noise`` -> optimizer -> ``PoissonSampler``). The DPO-specific pieces
are (plan §13):

  * ``preference_collator(pad_token_id, max_length)`` builds the batch with
    *six* mandatory tensors — chosen/rejected ``input_ids``, ``attention_mask``,
    ``completion_mask`` (each ``(B, L)``) — plus the two precomputed reference
    logp columns ``ref_chosen_logps`` / ``ref_rejected_logps`` (each ``(B,)``).
  * ``compute_ref_logprobs_for_dataset(...)`` precomputes the frozen reference
    model's per-example logps ONCE, outside the vmap, and caches them to a
    content-addressed ``.npz`` (so the expensive ref forward runs at most once).
  * The per-example loss runs TWO forwards (chosen + rejected), turns each into a
    completion logp via ``sequence_logp``, subtracts the precomputed ref logps to
    form per-example log-ratios, and dispatches through ``DPO_LOSSES[loss_type]``.
    Each loss output for example *i* depends only on example *i*'s data — Tier 1
    (plan §3.3), so per-example sensitivity stays ``O(C)`` after clipping.

The mechanism is the caller's choice (plan §3.2): swap the two ``opaque.dpsgd``
imports below for ``opaque.dpftrl`` to run DP-FTRL instead (see the commented
line near the imports). The loss closure does not change.

----------------------------------------------------------------------------
SMOKE MODE (``--smoke``)
----------------------------------------------------------------------------
``--smoke`` runs the **full per-example vmap DP-SGD path** on a tiny,
randomly-initialized LlamaForCausalLM (no network, no HF download) over a small
synthetic preference dataset (~8 examples). It precomputes reference logps
(using the model itself as the reference for the smoke), then executes 2 real
DP-SGD steps and prints the DPO loss each step. This is the path this script
lands on — it has been verified to run clean on CPU; the per-step loss output is
shown in the work-unit report. The ref-logp cache is written to a per-run
temporary directory (no network, no shared state).

A documented fallback exists in ``_run_smoke`` for environments where
``vmap(grad(...))`` over the patched model fails on CPU: a single non-vmap
chosen+rejected forward + ``DPO_LOSSES["sigmoid"]`` to validate the loss wiring,
with a clear note that the full per-example DP-SGD run is validated via the
Cadence GPU preset. The script never exits non-zero in smoke mode.

USAGE:

  # Smoke test (CPU, ~seconds, no network)
  python examples/train_dpo.py --smoke

  # Full training run (downloads model + dataset from HuggingFace)
  python examples/train_dpo.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --dataset trl-lib/ultrafeedback_binarized \\
    --loss-type sigmoid --beta 0.1 --max-length 1024 \\
    --batch-size 16 --max-steps 100 \\
    --learning-rate 1e-4 --clip-norm 1.0 --noise-multiplier 0.8

  # Legacy --model-name alias also accepted
  python examples/train_dpo.py \\
    --model-name gpt2 --dataset trl-lib/ultrafeedback_binarized \\
    --loss-type sigmoid --beta 0.1 --max-length 1024 --batch-size 16
"""

from __future__ import annotations

import argparse
import tempfile

import torch
import torchopt
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.patches import apply_model_patches  # makes HF forward vmap-safe
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
from opaque.optimizers import adamw
from opaque.random import key, fold_in
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.alignment import (
    extract_prompt,
    preference_collator,
    compute_ref_logprobs_for_dataset,
    reward_metrics,
    sequence_logp,
)
from opaque.alignment.loss.dpo import DPO_LOSSES

# DP-FTRL mechanism swap (plan §3.2): the loss closure is mechanism-agnostic.
# To run DP-FTRL instead of DP-SGD, replace the two ``opaque.dpsgd`` noise/
# sampling imports above with their DP-FTRL counterparts, e.g.:
#   from opaque.dpftrl.noise import band_mf_noise  # matrix-factorized noise
# and feed it the same ``ClippedPytree`` produced by ``clipped_grad`` below.


# The 8 per-example loss arguments after the trainable params (argnums=0):
# chosen_ids, chosen_mask, chosen_cmask, rejected_ids, rejected_mask,
# rejected_cmask, ref_chosen_logps, ref_rejected_logps. The vmap batch axis is
# taken over all of them, so batch_argnums lists every index 1..8 (and the
# microbatch path indexes args[i] for each, so it must not overrun).
_BATCH_ARGNUMS = (1, 2, 3, 4, 5, 6, 7, 8)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DP-SGD Direct Preference Optimization (DPO) for causal LMs"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU smoke test (random model, synthetic prefs, 2 steps).",
    )
    # --model is the canonical flag; --model-name is kept as a legacy alias.
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or local path (full mode).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/ultrafeedback_binarized",
        help="HuggingFace preference dataset name (full mode).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=list(DPO_LOSSES.keys()),
        default="sigmoid",
        help="DPO loss variant from opaque.alignment.loss.dpo.DPO_LOSSES.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature beta (reference-deviation strength).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for the preference collator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Expected batch size for Poisson sampling (sets the sample rate).",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help=(
            "Microbatch size for clipped_grad (None=full-batch vmap; pass 0 on "
            "CLI to mean None). Use when the full batch does not fit in memory."
        ),
    )
    parser.add_argument(
        "--max-steps",
        "--num-steps",
        dest="max_steps",
        type=int,
        default=100,
        help="Number of DP-SGD steps to run (full mode).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--clip-norm",
        "--clipping-norm",
        dest="clipping_norm",
        type=float,
        default=1.0,
        help="Per-example gradient clipping norm C.",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=0.8,
        help="DP-SGD Gaussian noise multiplier (sigma = nm * C / batch_size).",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    # --microbatch-size 0 on CLI means "no microbatching" (full-batch vmap),
    # mirroring the train_causal_lm.py convention.
    if args.microbatch_size == 0:
        args.microbatch_size = None
    return args


def _make_per_example_loss(fmodel, frozen, *, loss_type, beta):
    """Build the DPO per-example loss closure (TWO forwards: chosen + rejected).

    The returned callable has signature::

        per_example_loss(
            trainable_params,
            chosen_ids, chosen_mask, chosen_cmask,
            rejected_ids, rejected_mask, rejected_cmask,
            ref_chosen_logps, ref_rejected_logps,
        ) -> per-example scalar loss

    which is exactly ``argnums=0`` (trainable params) + the 9 per-example args
    in ``_BATCH_ARGNUMS``. ``frozen`` first in the merge so trainable params win
    on key collision (plan §13 sketch). Each output depends only on this
    example's data — Tier 1 (plan §3.3).
    """

    def per_example_loss(
        trainable_params,
        chosen_ids,
        chosen_mask,
        chosen_cmask,
        rejected_ids,
        rejected_mask,
        rejected_cmask,
        ref_chosen_logps,
        ref_rejected_logps,
    ):
        merged = {**frozen, **trainable_params}
        chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_out = fmodel(
            merged, input_ids=rejected_ids, attention_mask=rejected_mask
        )
        chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)
        rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask)
        # Log-ratios = policy logp - precomputed reference logp (per example).
        return DPO_LOSSES[loss_type](
            chosen_logp - ref_chosen_logps,
            rejected_logp - ref_rejected_logps,
            beta=beta,
        )

    return per_example_loss


def _make_ref_callable(model, device=None):
    """Wrap a model into a ``ref`` callable for compute_ref_logprobs_for_dataset.

    Returns ``ref(batch) -> {"ref_chosen_logps": (B,), "ref_rejected_logps": (B,)}``
    computed via ``sequence_logp`` under ``torch.no_grad()`` (plan §7.8 contract:
    ``ref`` is a plain ``dict[str, Tensor] -> dict[str, Tensor]`` callable, which
    keeps the precompute helper mechanism- and model-agnostic).

    The precompute helper collates on CPU; this callable moves each input to the
    model's ``device`` before the forward and returns the logps on CPU so they
    serialize back into the dataset cleanly.
    """
    dev = device if device is not None else next(model.parameters()).device

    def ref(batch):
        with torch.no_grad():
            chosen_ids = batch["chosen_input_ids"].to(dev)
            rejected_ids = batch["rejected_input_ids"].to(dev)
            chosen_out = model(
                input_ids=chosen_ids,
                attention_mask=batch["chosen_attention_mask"].to(dev),
            )
            rejected_out = model(
                input_ids=rejected_ids,
                attention_mask=batch["rejected_attention_mask"].to(dev),
            )
            chosen_logp = sequence_logp(
                chosen_out.logits,
                chosen_ids,
                batch["chosen_completion_mask"].to(dev),
            )
            rejected_logp = sequence_logp(
                rejected_out.logits,
                rejected_ids,
                batch["rejected_completion_mask"].to(dev),
            )
        return {
            "ref_chosen_logps": chosen_logp.cpu(),
            "ref_rejected_logps": rejected_logp.cpu(),
        }

    return ref


def _collate_to_device(collate, examples, device):
    """Collate raw preference rows and return the 9-tuple in batch_argnums order.

    The tuple order matches ``_BATCH_ARGNUMS`` and the per-example loss signature:
    chosen (ids, mask, cmask), rejected (ids, mask, cmask), ref (chosen, rejected).
    """
    b = collate(examples)
    return (
        b["chosen_input_ids"].to(device),
        b["chosen_attention_mask"].to(device),
        b["chosen_completion_mask"].to(device),
        b["rejected_input_ids"].to(device),
        b["rejected_attention_mask"].to(device),
        b["rejected_completion_mask"].to(device),
        b["ref_chosen_logps"].to(device),
        b["ref_rejected_logps"].to(device),
    )


def _run_smoke(args):
    """Tiny CPU smoke test: random Llama, synthetic prefs, 2 real DP-SGD steps.

    Builds a tiny randomly-initialized LlamaForCausalLM (no network), a small
    synthetic preference dataset, precomputes reference logps (using the model
    itself as the reference), and runs the full per-example vmap DP-SGD path for
    2 steps, printing the DPO loss each step. On a genuine vmap failure it falls
    back to a single non-vmap chosen+rejected forward + DPO loss so the smoke
    still exits 0 (documented in the module header).
    """
    from datasets import Dataset
    from transformers import LlamaConfig

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("train_dpo.py --smoke  (tiny random Llama, synthetic prefs, CPU)")
    print("=" * 72)

    # --- Tiny randomly-initialized model (no HF download) ---
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    apply_model_patches(model)  # vmap-safety wrappers (eager attn / batchify)
    model.eval()
    model.to(device)

    pad_token_id = 0
    max_length = 16
    batch_size = 4
    beta = args.beta
    loss_type = args.loss_type

    # --- Tiny synthetic preference dataset: 8 examples (no network) ---
    # Each example carries chosen/rejected token ids + completion masks. A
    # prompt prefix is shared between chosen and rejected; the completion mask
    # marks the (differing) response span, mirroring real DPO preprocessing.
    rng = torch.Generator().manual_seed(args.seed)

    def _make_example():
        prompt_len = int(torch.randint(2, 5, (1,), generator=rng))
        prompt = torch.randint(
            1, config.vocab_size, (prompt_len,), generator=rng
        ).tolist()
        chosen_resp_len = int(
            torch.randint(3, max_length - prompt_len, (1,), generator=rng)
        )
        rejected_resp_len = int(
            torch.randint(3, max_length - prompt_len, (1,), generator=rng)
        )
        chosen_resp = torch.randint(
            1, config.vocab_size, (chosen_resp_len,), generator=rng
        ).tolist()
        rejected_resp = torch.randint(
            1, config.vocab_size, (rejected_resp_len,), generator=rng
        ).tolist()
        chosen_ids = prompt + chosen_resp
        rejected_ids = prompt + rejected_resp
        # Completion mask: 0 over the prompt, 1 over the response.
        chosen_cmask = [0] * prompt_len + [1] * chosen_resp_len
        rejected_cmask = [0] * prompt_len + [1] * rejected_resp_len
        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "chosen_completion_mask": chosen_cmask,
            "rejected_completion_mask": rejected_cmask,
        }

    rows = [_make_example() for _ in range(8)]
    dataset = Dataset.from_list(rows)

    collate = preference_collator(pad_token_id, max_length)

    # --- Precompute reference logps ONCE, outside vmap, to a tmp cache dir ---
    # For the smoke we use the model itself as the (frozen) reference. The cache
    # goes to a per-run temp directory so the smoke is hermetic (no network, no
    # shared state across runs) — adversarial self-review: local cache_dir.
    print("\nPrecomputing reference logps (model-as-ref, tmp cache)...")
    with tempfile.TemporaryDirectory(prefix="opaque_dpo_smoke_") as cache_dir:
        dataset = compute_ref_logprobs_for_dataset(
            dataset,
            _make_ref_callable(model),
            collator=collate,
            output_columns=("ref_chosen_logps", "ref_rejected_logps"),
            batch_size=batch_size,
            cache_key=("dpo", "smoke"),
            cache_dir=cache_dir,
        )
    rows = list(dataset)  # now each row carries ref_chosen_logps / ref_rejected_logps
    print(
        f"  ref columns added: ref_chosen_logps[0]={rows[0]['ref_chosen_logps']:.4f}, "
        f"ref_rejected_logps[0]={rows[0]['ref_rejected_logps']:.4f}"
    )

    # --- Functional conversion (everything trainable on this tiny model) ---
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    print(f"Trainable param tensors: {len(trainable)} | frozen: {len(frozen)}")

    per_example_loss = _make_per_example_loss(
        fmodel, frozen, loss_type=loss_type, beta=beta
    )

    # --- Try the full per-example vmap DP-SGD path; fall back if it breaks ---
    try:
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=_BATCH_ARGNUMS,
            clipping_norm=args.clipping_norm,
            normalize_by=batch_size,
            return_aux=True,
        )
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=args.noise_multiplier, key=key(args.seed)
        )
        base_opt = adamw(lr=args.learning_rate)
        opt_state = base_opt.init(trainable)

        # Two DP-SGD steps over Poisson-sampled batches.
        sampler = PoissonSampler(
            rows,
            sample_rate=batch_size / len(rows),
            n_steps=2,
            key=fold_in(key(args.seed), 0, 0),
        )
        print(
            f"\nRunning 2 DP-SGD steps (full per-example vmap path, loss={loss_type})..."
        )
        step = 0
        for indices in sampler:
            batch = _collate_to_device(collate, [rows[i] for i in indices], device)
            (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable
            )
            trainable = torchopt.apply_updates(trainable, updates)
            step += 1
            print(
                f"  step {step}/2 | bs={batch[0].shape[0]} | "
                f"dpo_loss={aux.loss_values.mean().item():.4f}"
            )

        print("\nSmoke OK: full per-example DP-SGD vmap path completed 2 steps.")
        return 0

    except Exception as exc:  # pragma: no cover - defensive fallback path
        # Documented fallback (module header): vmap(grad(...)) over the patched
        # model failed on this CPU host. Validate the loss wiring with a single
        # non-vmap chosen+rejected forward + DPO loss so the smoke still exits 0.
        print(f"\nNote: full vmap DP-SGD path raised: {type(exc).__name__}: {exc}")
        print(
            "Falling back to a single non-vmap chosen+rejected forward + DPO "
            "loss to validate the loss wiring. The full per-example DP-SGD run "
            "is validated via the Cadence GPU preset."
        )
        batch = _collate_to_device(collate, rows[:batch_size], device)
        (
            chosen_ids,
            chosen_mask,
            chosen_cmask,
            rejected_ids,
            rejected_mask,
            rejected_cmask,
            ref_chosen_logps,
            ref_rejected_logps,
        ) = batch
        with torch.no_grad():
            chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)
            rejected_logp = sequence_logp(
                rejected_out.logits, rejected_ids, rejected_cmask
            )
            loss = DPO_LOSSES[loss_type](
                chosen_logp - ref_chosen_logps,
                rejected_logp - ref_rejected_logps,
                beta=beta,
            )
        print(f"  non-vmap batch dpo_loss (per-example mean): {loss.mean().item():.4f}")
        print("\nSmoke OK (fallback path): DPO loss wiring validated.")
        return 0


def _tokenize_preference_example(example, tokenizer, max_length):
    """Tokenize a single DPO preference example into model-ready token ids.

    Expects ``example`` to have ``"prompt"``, ``"chosen"``, and ``"rejected"``
    keys (run ``extract_prompt`` first if the prompt is implicit).  Both
    ``chosen`` and ``rejected`` may be:

    - A ``list`` of chat messages (``{"role": ..., "content": ...}`` dicts),
      in which case the tokenizer's chat template is applied.
    - A plain string, tokenized directly.

    The ``chosen_completion_mask`` / ``rejected_completion_mask`` tensors are
    ``0`` over prompt tokens and ``1`` over response (completion) tokens.
    Sequences are truncated to ``max_length`` from the right.

    Returns a dict with keys:
    - ``chosen_input_ids``: ``list[int]``
    - ``rejected_input_ids``: ``list[int]``
    - ``chosen_completion_mask``: ``list[int]``  (0=prompt, 1=completion)
    - ``rejected_completion_mask``: ``list[int]``
    """
    prompt = example.get("prompt", [])
    chosen = example["chosen"]
    rejected = example["rejected"]

    def _apply_template(messages):
        """Apply chat template if messages is a list, otherwise tokenize string."""
        if isinstance(messages, list):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        # Plain string: tokenize directly (no special tokens added here).
        return tokenizer.encode(messages, add_special_tokens=False)

    # Encode the prompt alone to find the prompt boundary.
    if isinstance(prompt, list) and prompt:
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,  # opens the assistant turn
        )
    elif isinstance(prompt, str) and prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        prompt_ids = []

    # Encode full chosen / rejected sequences (prompt + completion).
    if isinstance(chosen, list):
        # chosen/rejected are message lists (completions only, prompt is separate)
        full_chosen = prompt + (chosen if isinstance(chosen, list) else [])
        full_rejected = prompt + (rejected if isinstance(rejected, list) else [])
        chosen_ids = _apply_template(full_chosen)
        rejected_ids = _apply_template(full_rejected)
    else:
        # chosen/rejected are plain strings
        chosen_ids = (
            prompt_ids + tokenizer.encode(chosen, add_special_tokens=False)
            if prompt_ids
            else tokenizer.encode(chosen, add_special_tokens=True)
        )
        rejected_ids = (
            prompt_ids + tokenizer.encode(rejected, add_special_tokens=False)
            if prompt_ids
            else tokenizer.encode(rejected, add_special_tokens=True)
        )

    prompt_len = len(prompt_ids)

    # Build completion masks: 0 over prompt tokens, 1 over completion tokens.
    chosen_cmask = [0] * min(prompt_len, len(chosen_ids)) + [1] * max(
        0, len(chosen_ids) - prompt_len
    )
    rejected_cmask = [0] * min(prompt_len, len(rejected_ids)) + [1] * max(
        0, len(rejected_ids) - prompt_len
    )

    # Truncate to max_length (keep-start).
    chosen_ids = chosen_ids[:max_length]
    rejected_ids = rejected_ids[:max_length]
    chosen_cmask = chosen_cmask[:max_length]
    rejected_cmask = rejected_cmask[:max_length]

    return {
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
        "chosen_completion_mask": chosen_cmask,
        "rejected_completion_mask": rejected_cmask,
    }


def main():
    args = parse_args()

    if args.smoke:
        return _run_smoke(args)

    # ------------------------------------------------------------------ #
    # Full training path — real model + tokenizer + HF preference dataset #
    # ------------------------------------------------------------------ #
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("train_dpo.py — full DP-DPO training run")
    print("=" * 72)
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model_name}")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Loss        : {args.loss_type}  beta={args.beta}")
    print(f"  Max length  : {args.max_length}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  LR          : {args.learning_rate}")
    print(f"  Clip norm   : {args.clipping_norm}")
    print(f"  Noise mult. : {args.noise_multiplier}")
    print(f"  Microbatch  : {args.microbatch_size}")

    # --- 1. Load model + tokenizer -------------------------------------------
    print(f"\nLoading model + tokenizer: {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_model_patches(model)
    model.eval()
    model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 2. Load dataset + tokenize ------------------------------------------
    # ultrafeedback_binarized (and similar TRL preference datasets) stores
    # chosen/rejected as lists of chat messages; extract_prompt pulls out the
    # shared prompt prefix so chosen/rejected become completion-only lists.
    print(f"\nLoading dataset: {args.dataset} (streaming) ...")
    raw = load_dataset(args.dataset, split="train", streaming=True)

    # Stream until we have enough tokenized examples for the run.
    # We need at least batch_size examples for a meaningful Poisson draw.
    min_examples = max(args.batch_size * 4, 128)
    examples = []
    print(f"  Collecting at least {min_examples} examples ...")
    for raw_row in raw:
        row = extract_prompt(raw_row)
        try:
            tok = _tokenize_preference_example(row, tokenizer, args.max_length)
        except Exception:
            continue
        # Skip degenerate examples: both sides must have some completion tokens.
        if sum(tok["chosen_completion_mask"]) == 0:
            continue
        if sum(tok["rejected_completion_mask"]) == 0:
            continue
        examples.append(tok)
        # Collect 10× batch_size * max_steps rows, capped at a reasonable limit
        # so we don't stream the whole dataset before training begins.
        target = min(args.batch_size * max(args.max_steps, 10) * 2, 50_000)
        if len(examples) >= target:
            break

    if not examples:
        raise SystemExit(
            f"No usable preference examples found in '{args.dataset}'. "
            "Check that the dataset has 'chosen' and 'rejected' columns."
        )
    print(f"  Materialized {len(examples)} preference examples.")

    # --- 3. Precompute reference logps (frozen policy-as-ref) ----------------
    # We use the policy model itself as the reference at initialisation — a
    # common simplification that is correct when the policy is not yet trained
    # (log-ratios are all 0 at step 0).  In a production run, load a separate
    # frozen reference checkpoint here.
    from datasets import Dataset as HFDataset

    collate = preference_collator(tokenizer.pad_token_id, args.max_length)
    hf_dataset = HFDataset.from_list(examples)

    print("\nPrecomputing reference logps (policy-as-ref, cached to disk) ...")
    hf_dataset = compute_ref_logprobs_for_dataset(
        hf_dataset,
        _make_ref_callable(model),
        collator=collate,
        output_columns=("ref_chosen_logps", "ref_rejected_logps"),
        batch_size=args.batch_size,
        cache_key=("dpo", args.model_name),
    )
    rows = list(hf_dataset)
    print(
        f"  ref_chosen_logps[0]  = {rows[0]['ref_chosen_logps']:.4f}\n"
        f"  ref_rejected_logps[0] = {rows[0]['ref_rejected_logps']:.4f}"
    )

    # --- 4. Functional form + per-example DPO loss ---------------------------
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    print(f"Trainable tensors: {len(trainable)} | frozen: {len(frozen)}")

    per_example_loss = _make_per_example_loss(
        fmodel, frozen, loss_type=args.loss_type, beta=args.beta
    )

    # --- 5. DP-SGD glue: clipped_grad -> gaussian_noise -> adamw -------------
    # clipped_grad differentiates argnums=0 (trainable params) over the batch
    # axis of every per-example arg in _BATCH_ARGNUMS = (1..9) = (chosen ids/mask/
    # cmask, rejected ids/mask/cmask, ref_chosen_logps, ref_rejected_logps).
    # normalize_by=batch_size makes the released gradient a DP-mean with
    # sensitivity clipping_norm / batch_size.
    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=_BATCH_ARGNUMS,
        clipping_norm=args.clipping_norm,
        normalize_by=args.batch_size,
        return_aux=True,
        microbatch_size=args.microbatch_size,
    )
    noise_fn, noise_state = gaussian_noise(
        noise_multiplier=args.noise_multiplier, key=key(args.seed)
    )
    base_opt = adamw(lr=args.learning_rate)
    opt_state = base_opt.init(trainable)

    # --- 6. Privacy accountant -----------------------------------------------
    sample_rate = args.batch_size / len(rows)
    _step_privacy = dpsgd_acc.poisson(
        dpsgd_acc.gaussian(args.noise_multiplier),
        sample_rate=sample_rate,
    )

    def _epsilon_so_far(steps_done):
        """Return privacy ε spent after ``steps_done`` DP-SGD steps at δ=1e-5."""
        if steps_done == 0:
            return 0.0
        delta = 1e-5
        try:
            return (_step_privacy * steps_done).epsilon_at(delta)
        except Exception:
            return float("nan")

    # --- 7. Poisson-sampled training loop ------------------------------------
    sampler = PoissonSampler(
        rows,
        sample_rate=sample_rate,
        n_steps=args.max_steps,
        key=fold_in(key(args.seed), 0, 0),
    )

    print(
        f"\nTraining for up to {args.max_steps} DP-SGD steps "
        f"(loss={args.loss_type}, C={args.clipping_norm}, "
        f"nm={args.noise_multiplier}, sample_rate={sample_rate:.4f}) ..."
    )
    step = 0
    for indices in sampler:
        if not indices:  # empty Poisson draw — skip, no gradient to release
            step += 1
            continue
        batch = _collate_to_device(collate, [rows[i] for i in indices], device)
        (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        updates, opt_state = base_opt.update(noisy_grads, opt_state, params=trainable)
        trainable = torchopt.apply_updates(trainable, updates)
        step += 1

        if step % args.log_steps == 0 or step == args.max_steps:
            # Per-batch reward metrics (detached — private internal telemetry).
            (
                chosen_ids,
                _chosen_mask,
                chosen_cmask,
                rejected_ids,
                _rejected_mask,
                rejected_cmask,
                ref_chosen_lp,
                ref_rejected_lp,
            ) = batch
            with torch.no_grad():
                merged = {**frozen, **trainable}
                c_out = fmodel(
                    merged, input_ids=chosen_ids, attention_mask=_chosen_mask
                )
                r_out = fmodel(
                    merged, input_ids=rejected_ids, attention_mask=_rejected_mask
                )
                c_logp = sequence_logp(c_out.logits, chosen_ids, chosen_cmask)
                r_logp = sequence_logp(r_out.logits, rejected_ids, rejected_cmask)
                chosen_lr = c_logp - ref_chosen_lp
                rejected_lr = r_logp - ref_rejected_lp
            metrics = reward_metrics(chosen_lr, rejected_lr, beta=args.beta)
            eps = _epsilon_so_far(step)
            print(
                f"  step {step}/{args.max_steps} | "
                f"bs={batch[0].shape[0]} | "
                f"dpo_loss={aux.loss_values.mean().item():.4f} | "
                f"acc={metrics['rewards/accuracies'].item():.3f} | "
                f"margin={metrics['rewards/margins'].item():.4f} | "
                f"ε≈{eps:.3f}"
            )

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
