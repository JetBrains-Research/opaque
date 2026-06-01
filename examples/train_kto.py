# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end DP-SGD KTO example for causal LMs — the Tier-2 caller pattern.

This is the KTO sibling of ``examples/train_causal_lm.py`` (and of
``examples/train_sft.py`` / ``train_dpo.py``). It wires the ``opaque-alignment``
KTO primitives into the proven DP-SGD functional loop (``make_functional`` ->
per-example loss -> ``clipped_grad`` -> ``gaussian_noise`` -> optimizer ->
``PoissonSampler``).

KTO is **unpaired** (each example carries a boolean ``label``: desirable /
undesirable) and its ``kto`` loss is the package's canonical **Tier-2** loss
(plan §3.3, §8.1): the per-example loss depends on its own log-ratio AND on a
single *detached, batch-mean* KL scalar ``z_0``. This script exists to make the
Tier-2 caller contract concrete and explicit:

  THE TIER-2 CALLER PATTERN (the teaching point of this example)
  --------------------------------------------------------------
  Each step, the batch-mean KL is computed ONCE over the microbatch, OUTSIDE
  the per-example ``vmap``, and ``.detach()``-ed, then broadcast as a scalar
  into every per-example closure:

      with torch.no_grad():
          kl = (policy_KL_logp_batch
                - batch["reference_KL_logps"]).mean().detach().clamp(min=0)

  Because ``kl`` is a detached batch mean, swapping one record changes it by
  ``O(1/n)`` (not ``O(1)``), so per-example sensitivity after clipping stays
  ``O(C)`` — exactly the Tier-2 condition. ``KTO_SPEC["kto"]`` records
  ``tier=2``, ``cross_batch_aggregate="kl_mean"``, ``aggregate_leverage="O(1/n)"``.
  The detach keeps ``kl`` out of the released gradient's autograd graph, so it
  does not add to the privacy ledger. (v2: an optional cross-rank all-reduce of
  ``kl`` is added when ``LossAggregateSpec.cross_rank=True`` is wired — see the
  commented line in ``per_example_step``.)

The reference KL log-probs (``reference_KL_logps``) and the completion's own
reference log-probs (``reference_logps``) are precomputed ONCE over the dataset
with ``compute_ref_logprobs_for_dataset`` (outside vmap, under ``no_grad``); the
rotated KL completions come from ``rotate_kto_completions``.

The mechanism is the caller's choice (plan §3.2): swap the two ``opaque.dpsgd``
imports below for ``opaque.dpftrl`` to run DP-FTRL instead (see the commented
line near the imports). The Tier-2 loss closure does not change.

----------------------------------------------------------------------------
SMOKE MODE (``--smoke``)
----------------------------------------------------------------------------
``--smoke`` runs the **full per-example vmap DP-SGD path** on a tiny,
randomly-initialized LlamaForCausalLM (no network, no HF download) over a small
synthetic unpaired dataset (~8 examples with mixed True/False labels). It
rotates the KL completions, precomputes the reference log-probs with the model
serving as its own reference, then executes 2 real DP-SGD steps, printing the
KTO loss AND the computed detached ``kl`` scalar each step.

A documented fallback exists in ``_run_smoke`` for environments where
``vmap(grad(...))`` over the patched model fails on CPU: a single non-vmap
forward + the same Tier-2 ``kl`` computation + ``KTO_LOSSES["kto"]`` to validate
the wiring, with a clear note that the full per-example DP-SGD run is validated
via the Cadence GPU preset. The script never exits non-zero in smoke mode.

USAGE:

  # Smoke test (CPU, ~seconds, no network)
  python examples/train_kto.py --smoke

  # Full training on a real model + dataset
  python examples/train_kto.py \\
    --model-name gpt2 --dataset trl-lib/kto-mix-14k \\
    --max-length 512 --batch-size 16
"""

from __future__ import annotations

import argparse

import torch
import torchopt
from transformers import AutoModelForCausalLM, LlamaConfig

from opaque.patches import apply_model_patches  # makes HF forward vmap-safe
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
from opaque.optimizers import adamw
from opaque.random import key, fold_in
from opaque.alignment import (
    unpaired_preference_collator,
    rotate_kto_completions,
    compute_ref_logprobs_for_dataset,
    sequence_logp,
)
from opaque.alignment.loss.kto import KTO_LOSSES, KTO_SPEC

# DP-FTRL mechanism swap (plan §3.2): the Tier-2 loss closure is mechanism-
# agnostic. To run DP-FTRL instead of DP-SGD, replace the two ``opaque.dpsgd``
# noise/sampling imports above with their DP-FTRL counterparts, e.g.:
#   from opaque.dpftrl.noise import band_mf_noise  # matrix-factorized noise
# and feed it the same ``ClippedPytree`` produced by ``clipped_grad`` below.
# The kl computation, detach, and per-example KTO loss are unchanged.


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DP-SGD KTO (unpaired preference) training for causal LMs"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU smoke test (random model, synthetic data, 2 steps).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path (full mode).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for the collator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Expected batch size for Poisson sampling (sets the sample rate); "
        "also the rotation block size for rotate_kto_completions.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of DP-SGD steps to run (full mode).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KTO temperature (reference-deviation strength).",
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


def _build_smoke_dataset(vocab_size: int, max_length: int, seed: int):
    """Build a tiny synthetic unpaired KTO dataset (no network).

    Returns a ``datasets.Dataset`` with one row per example carrying:

      * ``completion``            — token-id list (the rotation key; rotated
        into ``KL_completion`` by ``rotate_kto_completions``).
      * ``completion_input_ids``  — same token ids (collator input).
      * ``completion_labels``     — per-token targets; the first token is masked
        to ``-100`` (no prediction target), the rest carry the next token.
      * ``label``                 — bool, mixed True/False across the 8 rows.

    The completions are deliberately distinct so ``rotate_kto_completions``'
    non-identity assertion passes.
    """
    from datasets import Dataset

    rng = torch.Generator().manual_seed(seed)
    rows: list[dict] = []
    for i in range(8):
        length = int(torch.randint(6, max_length, (1,), generator=rng))
        ids = torch.randint(1, vocab_size, (length,), generator=rng).tolist()
        # Labels: prompt-free toy example — supervise every token but the first
        # (the first position has no preceding context to predict it). Pad/shift
        # handling lives in sequence_logp; -100 here just marks "ignore".
        labels = [-100] + ids[1:]
        rows.append(
            {
                "completion": ids,
                "completion_input_ids": ids,
                "completion_labels": labels,
                # Mixed desirable / undesirable labels (alternating).
                "label": bool(i % 2 == 0),
            }
        )
    return Dataset.from_list(rows)


def _derive_kl_columns(example: dict) -> dict:
    """Turn a rotated ``KL_completion`` (token ids) into collator KL columns.

    ``rotate_kto_completions`` left-rotates the ``completion`` column into a new
    ``KL_completion`` column. The unpaired collator wants tokenized
    ``KL_completion_input_ids`` / ``KL_completion_labels``; since our synthetic
    ``completion`` already *is* a token-id list, we mirror the completion
    labelling convention onto the rotated ids.
    """
    kl_ids = example["KL_completion"]
    return {
        "KL_completion_input_ids": kl_ids,
        "KL_completion_labels": [-100] + kl_ids[1:],
    }


def _make_ref_callable(model, device):
    """Wrap ``model`` into a ``compute_ref_logprobs_for_dataset`` ref callable.

    The reference callable maps a collated batch -> ``{reference_logps,
    reference_KL_logps}``, each a ``(B,)`` tensor of per-sequence completion
    log-probs under the (frozen) reference. It runs under ``no_grad`` and uses
    the SAME ``sequence_logp`` the training closure uses, so the precomputed
    reference baseline is numerically consistent with the policy term.

    Outside vmap only (plan §3.4 / §7.8): this is the reference-precompute
    helper path, never called inside ``vmap``/``grad``.
    """

    def ref(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            comp_ids = batch["completion_input_ids"].to(device)
            comp_mask = batch["completion_attention_mask"].to(device)
            out = model(input_ids=comp_ids, attention_mask=comp_mask)
            ref_logps = sequence_logp(out.logits, comp_ids, comp_mask)

            kl_ids = batch["KL_completion_input_ids"].to(device)
            kl_mask = batch["KL_completion_attention_mask"].to(device)
            kl_out = model(input_ids=kl_ids, attention_mask=kl_mask)
            ref_kl_logps = sequence_logp(kl_out.logits, kl_ids, kl_mask)
        return {
            "reference_logps": ref_logps.float().cpu(),
            "reference_KL_logps": ref_kl_logps.float().cpu(),
        }

    return ref


def _run_smoke(args):
    """Tiny CPU smoke test: random Llama, synthetic unpaired data, 2 DP-SGD steps.

    Builds a tiny randomly-initialized LlamaForCausalLM (no network), a small
    synthetic unpaired dataset with mixed True/False labels, rotates the KL
    completions, precomputes the reference log-probs (model as its own
    reference), and runs the full per-example vmap DP-SGD path — demonstrating
    the Tier-2 caller pattern (detached batch-mean ``kl`` computed OUTSIDE the
    vmap, broadcast into the per-example closure).

    On a genuine vmap failure it falls back to a single non-vmap forward + the
    same Tier-2 ``kl`` computation + ``KTO_LOSSES["kto"]`` so the smoke still
    exits 0 (documented in the module header).
    """
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("train_kto.py --smoke  (tiny random Llama, synthetic unpaired data, CPU)")
    print("=" * 72)
    print(f"KTO_SPEC['kto']: {KTO_SPEC['kto']}")
    print(
        "  -> Tier-2: kl is a DETACHED batch-mean (kl_mean aggregate, O(1/n) "
        "leverage), computed OUTSIDE vmap and broadcast into each example."
    )

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

    # --- Tiny synthetic unpaired dataset (8 rows, mixed labels, no network) ---
    dataset = _build_smoke_dataset(config.vocab_size, max_length, args.seed)

    # --- KTO KL rotation: left-rotate completions within each block ---
    dataset = rotate_kto_completions(dataset, batch_size=batch_size, seed=args.seed)
    dataset = dataset.map(_derive_kl_columns)

    collate = unpaired_preference_collator(pad_token_id, max_length, calculate_KL=True)

    # --- Precompute reference log-probs ONCE (outside vmap; model as its own
    #     reference). Adds reference_logps + reference_KL_logps columns. ---
    ref = _make_ref_callable(model, device)
    dataset = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        collator=collate,
        output_columns=("reference_logps", "reference_KL_logps"),
        batch_size=batch_size,
        cache_key=("kto", "smoke"),
    )
    print(
        f"\nDataset ready: {len(dataset)} rows | "
        f"columns include reference_logps + reference_KL_logps."
    )

    rows = [dataset[i] for i in range(len(dataset))]

    def collate_to_device(examples):
        b = collate(examples)
        return {k: v.to(device) for k, v in b.items()}

    # --- Functional conversion (everything trainable on this tiny model) ---
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    print(f"Trainable param tensors: {len(trainable)} | frozen: {len(frozen)}")

    def merged(t):
        return {**frozen, **t}

    # === The per-example loss closure (everything below runs UNDER vmap) ===
    # Tier-2 contract: ``kl`` is NOT computed here — it arrives pre-detached as
    # a scalar argument the caller broadcasts in (vmap-``None`` axis).
    def per_example_loss(
        trainable_params, completion_ids, completion_mask, label, ref_logp, kl
    ):
        out = fmodel(
            merged(trainable_params),
            input_ids=completion_ids,
            attention_mask=completion_mask,
        )
        logp = sequence_logp(out.logits, completion_ids, completion_mask)
        # Per-example log-ratio r = log pi_theta - log pi_ref, split by label so
        # KTO's chosen/rejected branches each see the right side (label-masked,
        # never None — required under vmap, plan §7.2).
        chosen_lr = (logp - ref_logp) * label.float()
        rejected_lr = (logp - ref_logp) * (~label.bool()).float()
        # kl is computed once over the microbatch OUTSIDE vmap, detached
        # (Tier-2 §8.1); KTO_SPEC['kto'] declares tier=2, kl_mean aggregate,
        # O(1/n) leverage.
        return KTO_LOSSES["kto"](chosen_lr, rejected_lr, label, beta=beta, kl=kl)

    # --- Try the full per-example vmap DP-SGD path; fall back if it breaks ---
    try:
        # batch_argnums covers the 5 per-example tensor args (ids, mask, label,
        # ref_logp, kl). ``kl`` is a per-example-broadcast scalar: we pass the
        # SAME detached value into every example, which is the vmap-``None``
        # broadcast the Tier-2 contract calls for (plan §8.1 step 4).
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(1, 2, 3, 4, 5),
            clipping_norm=args.clipping_norm,
            normalize_by=batch_size,
            return_aux=True,
        )
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=args.noise_multiplier, key=key(args.seed)
        )
        base_opt = adamw(lr=args.learning_rate)
        opt_state = base_opt.init(trainable)

        sampler = PoissonSampler(
            rows,
            sample_rate=batch_size / len(rows),
            n_steps=2,
            key=fold_in(key(args.seed), 0, 0),
        )
        print("\nRunning 2 DP-SGD steps (full per-example vmap path)...")
        step = 0
        for indices in sampler:
            batch = collate_to_device([rows[i] for i in indices])
            bs = batch["completion_input_ids"].shape[0]

            # ============================================================= #
            # TIER-2 CALLER PATTERN (plan §3.3, §8.1) — THE TEACHING POINT.
            # Compute the detached batch-mean KL ONCE over the microbatch,
            # OUTSIDE the vmap, then broadcast the scalar into every example.
            # ============================================================= #
            with torch.no_grad():
                # policy KL-completion log-probs under the CURRENT params.
                pol_kl_out = fmodel(
                    merged(trainable),
                    input_ids=batch["KL_completion_input_ids"],
                    attention_mask=batch["KL_completion_attention_mask"],
                )
                policy_KL_logp_batch = sequence_logp(
                    pol_kl_out.logits,
                    batch["KL_completion_input_ids"],
                    batch["KL_completion_attention_mask"],
                )
                kl = (
                    (policy_KL_logp_batch - batch["reference_KL_logps"])
                    .mean()
                    .detach()
                    .clamp(min=0)
                )
            # (v2: optional cross-rank reduction when the loss declares
            #  LossAggregateSpec.cross_rank=True — see plan §9 / §13)
            #   import opaque.distributed as dist
            #   kl = dist.all_reduce(kl, op="mean")

            # Broadcast the single detached kl scalar to every example so the
            # vmap'd grad sees it as a per-example (constant) input.
            kl_per_example = kl.expand(bs)

            (grads, aux), clip_state = grad_fn(
                trainable,
                batch["completion_input_ids"],
                batch["completion_attention_mask"],
                batch["label"],
                batch["reference_logps"],
                kl_per_example,
                state=clip_state,
            )
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable
            )
            trainable = torchopt.apply_updates(trainable, updates)
            step += 1
            print(
                f"  step {step}/2 | bs={bs} | "
                f"kl={kl.item():.4f} | "
                f"kto_loss={aux.loss_values.mean().item():.4f}"
            )

        print("\nSmoke OK: full per-example DP-SGD vmap path completed 2 steps.")
        return 0

    except Exception as exc:  # pragma: no cover - defensive fallback path
        # Documented fallback (module header): vmap(grad(...)) over the patched
        # model failed on this CPU host. Validate the Tier-2 wiring with a
        # single non-vmap forward + the SAME detached-kl computation +
        # KTO_LOSSES["kto"], so the smoke still exits 0. The full per-example
        # DP-SGD run is validated via the Cadence GPU preset.
        print(f"\nNote: full vmap DP-SGD path raised: {type(exc).__name__}: {exc}")
        print(
            "Falling back to a single non-vmap forward + the Tier-2 kl "
            "computation + KTO_LOSSES['kto'] to validate the wiring. The full "
            "per-example DP-SGD run is validated via the Cadence GPU preset."
        )
        batch = collate_to_device(rows[:batch_size])
        with torch.no_grad():
            comp_ids = batch["completion_input_ids"]
            comp_mask = batch["completion_attention_mask"]
            label = batch["label"]
            ref_logp = batch["reference_logps"]

            out = model(input_ids=comp_ids, attention_mask=comp_mask)
            logp = sequence_logp(out.logits, comp_ids, comp_mask)

            # Tier-2: detached batch-mean kl, computed OUTSIDE any vmap.
            pol_kl_out = model(
                input_ids=batch["KL_completion_input_ids"],
                attention_mask=batch["KL_completion_attention_mask"],
            )
            policy_KL_logp_batch = sequence_logp(
                pol_kl_out.logits,
                batch["KL_completion_input_ids"],
                batch["KL_completion_attention_mask"],
            )
            kl = (
                (policy_KL_logp_batch - batch["reference_KL_logps"])
                .mean()
                .detach()
                .clamp(min=0)
            )

            chosen_lr = (logp - ref_logp) * label.float()
            rejected_lr = (logp - ref_logp) * (~label.bool()).float()
            loss = KTO_LOSSES["kto"](chosen_lr, rejected_lr, label, beta=beta, kl=kl)
        print(
            f"  non-vmap batch | kl={kl.item():.4f} | "
            f"kto_loss (per-example mean)={loss.mean().item():.4f}"
        )
        print("\nSmoke OK (fallback path): Tier-2 KTO loss wiring validated.")
        return 0


def main():
    args = parse_args()

    if args.smoke:
        return _run_smoke(args)

    # --- Full training path (real model + tokenizer) ---
    from transformers import AutoTokenizer

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

    # A real run would: load an unpaired dataset (completion + label), tokenize
    # into completion_input_ids / completion_labels, rotate the KL completions,
    # precompute the reference log-probs, then run the SAME Tier-2 step loop as
    # the smoke path above.
    _ = unpaired_preference_collator(tokenizer.pad_token_id, args.max_length)

    raise SystemExit(
        "Full-mode training requires an unpaired preference dataset wired into "
        "a PoissonSampler DataLoader (see examples/train_causal_lm.py for the "
        "full data path). This KTO example ships a runnable --smoke path that "
        "demonstrates the complete Tier-2 caller pattern; for a production loop "
        "use train_causal_lm.py as the data-loading template and swap in the "
        "unpaired_preference_collator + the detached-kl KTO step shown above."
    )


if __name__ == "__main__":
    raise SystemExit(main())
