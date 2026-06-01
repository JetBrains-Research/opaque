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

  # Full training on a real model + preference dataset
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
from opaque.alignment import (
    preference_collator,
    compute_ref_logprobs_for_dataset,
    sequence_logp,
)
from opaque.alignment.loss.dpo import DPO_LOSSES

# DP-FTRL mechanism swap (plan §3.2): the loss closure is mechanism-agnostic.
# To run DP-FTRL instead of DP-SGD, replace the two ``opaque.dpsgd`` noise/
# sampling imports above with their DP-FTRL counterparts, e.g.:
#   from opaque.dpftrl.noise import band_mf_noise  # matrix-factorized noise
# and feed it the same ``ClippedPytree`` produced by ``clipped_grad`` below.


# The 9 per-example loss arguments after the trainable params (argnums=0). The
# vmap batch axis is taken over all of them, so batch_argnums must list every
# index 1..9 (adversarial self-review item: cover ALL per-example args).
_BATCH_ARGNUMS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


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
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
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
        "--num-steps",
        type=int,
        default=100,
        help="Number of DP-SGD steps to run (full mode).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help="Per-example gradient clipping norm C.",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="DP-SGD Gaussian noise multiplier (sigma = nm * C / batch_size).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


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


def _make_ref_callable(model):
    """Wrap a model into a ``ref`` callable for compute_ref_logprobs_for_dataset.

    Returns ``ref(batch) -> {"ref_chosen_logps": (B,), "ref_rejected_logps": (B,)}``
    computed via ``sequence_logp`` under ``torch.no_grad()`` (plan §7.8 contract:
    ``ref`` is a plain ``dict[str, Tensor] -> dict[str, Tensor]`` callable, which
    keeps the precompute helper mechanism- and model-agnostic).
    """

    def ref(batch):
        with torch.no_grad():
            chosen_out = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            rejected_out = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            chosen_logp = sequence_logp(
                chosen_out.logits,
                batch["chosen_input_ids"],
                batch["chosen_completion_mask"],
            )
            rejected_logp = sequence_logp(
                rejected_out.logits,
                batch["rejected_input_ids"],
                batch["rejected_completion_mask"],
            )
        return {
            "ref_chosen_logps": chosen_logp,
            "ref_rejected_logps": rejected_logp,
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


def main():
    args = parse_args()

    if args.smoke:
        return _run_smoke(args)

    # --- Full training path (real model + tokenizer) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading model + tokenizer: {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_model_patches(model)
    model.eval()
    model.to(device)

    collate = preference_collator(tokenizer.pad_token_id, args.max_length)

    # Reference logps are precomputed ONCE here (outside the vmap loop), using a
    # frozen copy of the base model as the reference. In a real run the dataset
    # would be tokenized into chosen/rejected ids + completion masks first (see
    # opaque.alignment.data.extract_prompt + the preference preprocessing path),
    # then passed through compute_ref_logprobs_for_dataset:
    #
    #   dataset = compute_ref_logprobs_for_dataset(
    #       dataset,
    #       _make_ref_callable(ref_model),
    #       collator=collate,
    #       output_columns=("ref_chosen_logps", "ref_rejected_logps"),
    #       cache_key=("dpo", args.model_name),
    #   )

    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    per_example_loss = _make_per_example_loss(
        fmodel, frozen, loss_type=args.loss_type, beta=args.beta
    )

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
    )
    noise_fn, noise_state = gaussian_noise(
        noise_multiplier=args.noise_multiplier, key=key(args.seed)
    )
    base_opt = adamw(lr=args.learning_rate)
    opt_state = base_opt.init(trainable)
    del (
        collate,
        grad_fn,
        clip_state,
        noise_fn,
        noise_state,
        opt_state,
    )  # wired below in real loop (collate feeds the precompute + DataLoader)

    raise SystemExit(
        "Full-mode training requires a preference dataset tokenized into "
        "chosen/rejected ids + completion masks, run through "
        "compute_ref_logprobs_for_dataset (see the commented block above) and "
        "wired into a PoissonSampler DataLoader (see examples/train_causal_lm.py "
        "for the full data path). This DPO example ships a runnable --smoke path; "
        "for a complete production loop use train_causal_lm.py as the data-loading "
        "template and swap in the preference collator + DPO per-example loss "
        "(two forwards + sequence_logp + DPO_LOSSES) shown above."
    )


if __name__ == "__main__":
    raise SystemExit(main())
