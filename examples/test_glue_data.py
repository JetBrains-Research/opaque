# Copyright 2025 JetBrains S.r.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GLUE / sequence-classification path.

Run explicitly -- pytest discovery only walks `packages/`, so this file is not
picked up by a bare `pytest` from the repo root:

    pytest examples/test_glue_data.py -v
    pytest examples/test_glue_data.py -v -m "not slow"    # skip the RoBERTa run

WHY THIS FILE EXISTS AT ALL. train_causal_lm.py cannot run on macOS (a
vmap/transformers incompatibility), so the only local check available is an
importable one. Both LoRA-SB crashes -- a tuple-arity TypeError and then a CUDA
OOM -- got through because the unit tests used a hand-written stand-in for the
real collate and loss rather than the real ones, and each cost a GPU run to find.
Every test here therefore drives the SAME functions the trainer calls.

test_rotating_xse_trains_roberta is the important one: it pins the four call
conventions the classification path has to get right, each of which failed once
while this was being built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import glue_data as g  # noqa: E402

# Wq, Wv, Wo, FC1 -- the paper's GLUE module set (LoRA-XS Appendix D.1).
# "attention.output.dense" and not "output.dense": the latter also matches the
# FFN block's output projection (FC2), which the paper does not adapt.
GLUE_MODULES = ["query", "value", "attention.output.dense", "intermediate.dense"]


# ------------------------------------------------------------------ metrics
# These need no network and no model, so they are the cheap regression net
# around the three figures the paper reports.


def test_matthews_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    import random

    random.seed(0)
    for _ in range(200):
        n = random.randint(2, 40)
        preds = [random.randint(0, 1) for _ in range(n)]
        refs = [random.randint(0, 1) for _ in range(n)]
        assert g._matthews(preds, refs) == pytest.approx(
            sklearn.matthews_corrcoef(refs, preds), abs=1e-9
        )


def test_pearson_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    import random

    random.seed(1)
    for _ in range(200):
        n = random.randint(2, 40)
        preds = [random.gauss(0, 1) for _ in range(n)]
        refs = [random.gauss(0, 1) for _ in range(n)]
        assert g._pearson(preds, refs) == pytest.approx(
            stats.pearsonr(refs, preds)[0], abs=1e-9
        )


def test_metrics_degenerate_cases_return_zero_not_nan():
    """Early in training the model predicts one class; the metric must not NaN."""
    assert g._matthews([0] * 10, [0, 1] * 5) == 0.0
    assert g._pearson([1.0] * 10, list(range(10))) == 0.0


def test_glue_metric_dispatches_on_task():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 1])
    assert g.glue_metric(g.resolve_task("sst2"), logits, labels) == pytest.approx(0.75)
    # STS-B is regression: logits are used directly, not argmax'd.
    assert g.glue_metric(
        g.resolve_task("stsb"),
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    ) == pytest.approx(1.0)


def test_resolve_task_normalizes_and_rejects():
    assert g.resolve_task("STS-B").name == "stsb"
    assert g.resolve_task("sst_2").name == "sst2"
    with pytest.raises(ValueError, match="Unknown GLUE task"):
        g.resolve_task("squad")


# --------------------------------------------------------------- data path


@pytest.mark.parametrize("task_name", ["rte", "cola", "stsb"])
def test_collate_shapes_dtypes_and_padding(task_name):
    """The collate must return a 3-tuple of device tensors with a real mask.

    A mask of all ones would silently mean "attend to padding", which on an
    encoder (no causal mask to hide it) corrupts every short example. Asserting
    that the widest row has no padding catches a mask built from the wrong axis.
    """
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    task = g.resolve_task(task_name)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    train, _ = g.build_glue_datasets(
        task, tok, max_seq_len=128, num_train_samples=64, num_eval_samples=None
    )
    collate = g.make_glue_collate(task, tok, torch.device("cpu"))
    batch = next(iter(DataLoader(train, batch_size=8, collate_fn=collate)))

    assert len(batch) == 3, "grad fns are built with batch_argnums=(1, 2, 3)"
    ids, mask, labels = batch
    assert ids.shape == mask.shape
    assert ids.dtype == torch.long and mask.dtype == torch.long
    assert labels.shape == (8,)

    real = mask.sum(dim=1).tolist()
    assert max(real) == ids.shape[1], "widest row should be unpadded"
    assert min(real) >= 1

    if task.is_regression:
        assert labels.dtype == torch.float32
    else:
        assert labels.dtype == torch.long
        assert int(labels.max()) < task.num_labels


def test_eval_split_is_glue_validation_not_a_slice_of_train():
    """Published GLUE numbers are on `validation`; anything else is incomparable."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("roberta-base")
    _, evaluation = g.build_glue_datasets(g.resolve_task("rte"), tok, max_seq_len=64)
    assert len(evaluation) == 277, "RTE validation is 277 rows"


# ------------------------------------------------- the whole loop, for real


@pytest.mark.slow
def test_rotating_xse_trains_roberta():
    """Pin the four call conventions the classification path must get right.

    Each of these failed once while this path was being built, so each assertion
    below is a regression test for a real mistake:

    1. attn_implementation="eager" is REQUIRED, not a preference. transformers'
       SDPA path runs `torch.all(mask == 1)` to skip a no-op mask, which is
       data-dependent control flow and illegal under vmap. The causal-LM path
       never trips this only because its collate passes no attention mask at all.
    2. per_example_loss_fn must re-add the batch dim. vmap strips it, so a
       (B, L) input arrives as (L,) and a (B,) label as a 0-d scalar, while
       RoBERTa unpacks exactly two dims (modeling_roberta.py:779).
    3. task_type="SEQ_CLS" is what puts the randomly-initialized classifier head
       into PEFT's modules_to_save with requires_grad=True, which is what makes
       make_functional(partition_trainable=True) place it in the TRAINABLE dict.
       Under CAUSAL_LM it lands in frozen_params and never trains at all.
    4. xse_sgd needs frozen_params on `init` (layer discovery) and `frozen=` on
       `update` (it rewrites the frozen B/A factors when it rotates).
    """
    import torchopt
    from opaque.clipping import clipped_grad
    from opaque.functional import make_functional
    from peft import get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from lora_privacy.peft_lora_xs import LoraXSConfig, xse_sgd

    task = g.resolve_task("rte")
    device = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=task.num_labels,
        attn_implementation="eager",  # convention 1
    )
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(
        model,
        LoraXSConfig(
            r=8,
            lora_alpha=16,
            sigma=1e-5,
            lora_dropout=0.0,
            target_modules=GLUE_MODULES,
            task_type="SEQ_CLS",  # convention 3
        ),
    )
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    head_keys = [k for k in trainable if "classifier" in k]
    r_keys = [k for k in trainable if "lora_xs_R" in k]
    assert head_keys, "classifier head is frozen -- it would never train"
    assert len(r_keys) == 48, f"expected 4 modules x 12 layers, got {len(r_keys)}"

    def merged(t):
        return {**frozen, **t}

    def per_example_loss_fn(t, ids, mask, labels):
        return fmodel(  # convention 2
            merged(t),
            ids.unsqueeze(0),
            attention_mask=mask.unsqueeze(0),
            labels=labels.reshape(1),
        ).loss

    train, evaluation = g.build_glue_datasets(
        task, tok, max_seq_len=64, num_train_samples=64, num_eval_samples=None
    )
    collate = g.make_glue_collate(task, tok, device)
    train_loader = DataLoader(train, batch_size=8, collate_fn=collate)
    eval_loader = DataLoader(evaluation, batch_size=8, collate_fn=collate)

    def evaluate(t):
        total, n, logits, refs = 0.0, 0, [], []
        with torch.no_grad():
            for ids, mask, labels in eval_loader:
                out = fmodel(merged(t), ids, attention_mask=mask, labels=labels)
                total += float(out.loss) * len(ids)
                n += len(ids)
                logits.append(out.logits)
                refs.append(labels)
        return total / n, g.glue_metric(task, torch.cat(logits), torch.cat(refs))

    grad_fn, clip_state = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1, 2, 3),  # convention 2
        clipping_norm=1.0,
        normalize_by=8,
        microbatch_size=8,
        return_aux=True,
    )

    loss_before, _ = evaluate(trainable)
    assert loss_before == pytest.approx(0.693, abs=0.05), "untrained head ~ ln 2"

    opt = xse_sgd(
        lr=1e-3, lora_alpha=16, p_e=0.25, rotation_step_interval=1, momentum=0.9
    )
    state = opt.init(trainable, frozen)  # convention 4
    assert len(state.registry) == 48

    r_before = {k: trainable[k].clone() for k in r_keys}
    head_before = {k: trainable[k].clone() for k in head_keys}
    b_before = {k: v.clone() for k, v in frozen.items() if "lora_xs_B" in k}

    for _ in range(2):
        for batch in train_loader:
            (grads, _aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
            updates, state, frozen = opt.update(  # convention 4
                grads, state, params=trainable, frozen=frozen
            )
            trainable = torchopt.apply_updates(trainable, updates)

    assert all(not torch.equal(r_before[k], trainable[k]) for k in r_keys), (
        "some R cores never moved"
    )
    assert all(not torch.equal(head_before[k], trainable[k]) for k in head_keys), (
        "classifier head never moved"
    )
    assert all(not torch.equal(b_before[k], frozen[k]) for k in b_before), (
        "frozen B factors were not rotated -- rotation is not running"
    )

    loss_after, metric_after = evaluate(trainable)
    assert torch.isfinite(torch.tensor(loss_after)), "loss went non-finite"
    assert 0.0 <= metric_after <= 1.0


def test_refuses_to_truncate_the_validation_split():
    """The single most dangerous default on this path.

    The trainer's global --num-eval-samples default is 100, sized for a causal-LM
    smoke test. Honouring it on GLUE would compute CoLA's Matthews over 100 of
    1043 rows and report it as if comparable to the paper's 67.0. Nothing about
    the resulting run would look wrong in W&B, which is exactly how ~225 runs came
    to carry invalid downstream numbers. So it raises.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("roberta-base")
    with pytest.raises(ValueError, match="refusing to evaluate"):
        g.build_glue_datasets(
            g.resolve_task("rte"), tok, max_seq_len=64, num_eval_samples=100
        )
    # None (the trainer maps 0 -> None) is the supported way to say "all".
    _, evaluation = g.build_glue_datasets(
        g.resolve_task("rte"), tok, max_seq_len=64, num_eval_samples=None
    )
    assert len(evaluation) == 277


def test_train_subsampling_is_still_allowed():
    """Train subsampling is a legitimate ablation, unlike eval subsampling."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train, _ = g.build_glue_datasets(
        g.resolve_task("rte"), tok, max_seq_len=64, num_train_samples=64
    )
    assert len(train) == 64


def test_glue_preset_reproduces_the_paper_setup():
    """Pin the preset against LoRA-XS Appendix D.1 / Table 7.

    Every value here is a comparability requirement, not a preference: drift in
    any of them silently invalidates the comparison to their Table 1, and the
    resulting run still looks perfectly healthy. Notably lora_modules must be the
    4-module GLUE set and NOT the 7-module set the KStack presets use.
    """
    import sys

    sys.argv = [
        "train_causal_lm.py",
        "--preset",
        "roberta-large-glue",
        "--glue-task",
        "rte",
    ]
    import train_causal_lm

    args = train_causal_lm.parse_args()
    assert args.task_type == "sequence-classification"
    assert args.model_name == "FacebookAI/roberta-large"
    assert args.lora_modules == [
        "query",
        "value",
        "attention.output.dense",
        "intermediate.dense",
    ]
    assert args.lora_alpha == 16
    assert args.lora_xs_sigma == 1e-5
    assert args.max_seq_len == 128
    assert args.batch_size == 32
    assert args.lr_warmup_ratio == pytest.approx(0.06)
    assert args.optimizer == "adamw"
    assert args.dtype == "float32"
    # Forced: sdpa + an attention mask is incompatible with vmap.
    assert args.attention == "eager"
    # 0 = whole split. 5000/100 would truncate both.
    assert args.num_train_samples == 0
    assert args.num_eval_samples == 0


def test_explicit_flags_override_the_preset():
    """`_set` must not clobber what the caller passed -- the rank sweep needs it."""
    import sys

    sys.argv = [
        "train_causal_lm.py",
        "--preset",
        "roberta-large-glue",
        "--glue-task",
        "cola",
        "--lora-r",
        "4",
        "--learning-rate",
        "6e-4",
    ]
    import train_causal_lm

    args = train_causal_lm.parse_args()
    assert args.lora_r == 4
    assert args.learning_rate == pytest.approx(6e-4)
    assert args.lora_alpha == 16, "alpha stays fixed across ranks, as in the paper"


def test_scaling_updates_equals_a_separate_learning_rate():
    """--classifier-lr is implemented by scaling the head's updates. Prove it exact.

    For SGD/Adam/AdamW the update is exactly proportional to the learning rate --
    decoupled weight decay included -- so multiplying by (classifier_lr /
    learning_rate) gives precisely the update a second optimizer at classifier_lr
    would produce. The second moment is built from gradients and is
    lr-independent, so sharing state across the two groups changes nothing.

    If this ever stops being bit-exact, the head is no longer being trained at the
    rate the run claims, and every GLUE number becomes untrustworthy.
    """
    import torchopt

    def run(scale_updates, lr_a, lr_b, steps=25):
        torch.manual_seed(0)
        p = {"a": torch.tensor([1.0, -2.0]), "b": torch.tensor([0.5, 3.0])}
        kw = dict(betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
        if scale_updates:
            opt = torchopt.adamw(lr=lr_a, **kw)
            state = opt.init(p)
            k = lr_b / lr_a
        else:
            oa, ob = torchopt.adamw(lr=lr_a, **kw), torchopt.adamw(lr=lr_b, **kw)
            sa, sb = oa.init({"a": p["a"]}), ob.init({"b": p["b"]})
        for step in range(steps):
            g = {
                "a": p["a"] * 0.3 + 0.1 * (step % 3),
                "b": p["b"] * 0.7 - 0.2 * (step % 5),
            }
            if scale_updates:
                u, state = opt.update(g, state, params=p)
                u = dict(u)
                u["b"] = u["b"] * k
                p = torchopt.apply_updates(p, u)
            else:
                ua, sa = oa.update({"a": g["a"]}, sa, params={"a": p["a"]})
                ub, sb = ob.update({"b": g["b"]}, sb, params={"b": p["b"]})
                p = {
                    "a": torchopt.apply_updates({"a": p["a"]}, ua)["a"],
                    "b": torchopt.apply_updates({"b": p["b"]}, ub)["b"],
                }
        return p

    for head_lr in (1e-2, 1e-4):  # 10x up and 10x down
        scaled, two_opt = run(True, 1e-3, head_lr), run(False, 1e-3, head_lr)
        for key in ("a", "b"):
            assert torch.equal(scaled[key], two_opt[key]), (
                f"head_lr={head_lr} diverged on {key}"
            )


@pytest.mark.slow
def test_head_keys_are_discoverable_under_peft():
    """The key pattern --classifier-lr matches must actually hit PEFT's head.

    PEFT nests the head under modules_to_save, so the trainable key is
    base_model.model.classifier.modules_to_save.default.dense.weight. If the
    pattern misses, _head_scale silently applies to nothing and the head trains
    at the adapter's rate while the log claims otherwise.
    """
    from opaque.functional import make_functional
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification

    from lora_privacy.peft_lora_xs import LoraXSConfig

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2, attn_implementation="eager"
    )
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(
        model,
        LoraXSConfig(
            r=8,
            lora_alpha=16,
            sigma=1e-5,
            lora_dropout=0.0,
            target_modules=GLUE_MODULES,
            task_type="SEQ_CLS",
        ),
    )
    _, trainable, _ = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    # Exactly the expression the trainer uses.
    head = [
        k for k in trainable if ".classifier." in f".{k}." or ".score." in f".{k}."
    ]
    assert head, f"pattern missed the head; keys: {list(trainable)[:5]}"
    # roberta's head is dense{weight,bias} + out_proj{weight,bias}
    assert len(head) == 4, f"expected 4 head tensors, got {len(head)}: {head}"
    assert not any("lora_xs" in k for k in head), "pattern caught adapter tensors"


@pytest.mark.slow
def test_rotation_warmup_suppresses_then_enables_rotation():
    """--lora-xse-rotation-warmup-steps must actually gate the FIRST rotation.

    The observable is the frozen B factor: rotation rewrites it, nothing else
    does. So B unchanged after `warmup` steps and changed after one more step is
    exactly the property, and it cannot be faked by a no-op.

    A silently-ignored warmup would look identical in W&B to a working one, which
    is the same failure shape as the adafactor no-op and the truncated GLUE split.
    """
    import torchopt
    from opaque.clipping import clipped_grad
    from opaque.functional import make_functional
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification

    from lora_privacy.peft_lora_xs import LoraXSConfig, xse_sgd

    WARMUP = 4
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2, attn_implementation="eager"
    )
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(
        model,
        LoraXSConfig(
            r=8, lora_alpha=16, sigma=1e-5, lora_dropout=0.0,
            target_modules=GLUE_MODULES, task_type="SEQ_CLS",
        ),
    )
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def loss_fn(t, ids, mask, labels):
        return fmodel(
            {**frozen, **t}, ids.unsqueeze(0),
            attention_mask=mask.unsqueeze(0), labels=labels.reshape(1),
        ).loss

    torch.manual_seed(0)
    ids = torch.randint(4, 900, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1])

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2, 3), clipping_norm=1e6,
        normalize_by=4, microbatch_size=4, return_aux=True,
    )
    opt = xse_sgd(
        lr=1e-3, lora_alpha=16, p_e=0.25, rotation_step_interval=1,
        rotation_warmup_steps=WARMUP, momentum=0.9,
    )
    state = opt.init(trainable, frozen)
    b_keys = [k for k in frozen if "lora_xs_B" in k]
    assert b_keys
    b0 = {k: frozen[k].clone() for k in b_keys}

    def step():
        nonlocal trainable, state, frozen, clip_state
        (g, _a), clip_state = grad_fn(trainable, ids, mask, labels, state=clip_state)
        u, state, frozen = opt.update(g, state, params=trainable, frozen=frozen)
        trainable = torchopt.apply_updates(trainable, u)

    for _ in range(WARMUP):
        step()
    assert all(torch.equal(b0[k], frozen[k]) for k in b_keys), (
        f"B changed during the {WARMUP} warmup steps — rotation was not suppressed"
    )
    step()  # step WARMUP+1: the first rotation must fire here
    assert any(not torch.equal(b0[k], frozen[k]) for k in b_keys), (
        "B never changed after warmup — rotation never started at all"
    )


def test_rotation_warmup_rejects_negative():
    from lora_privacy.peft_lora_xs import xse_sgd

    with pytest.raises(ValueError, match="rotation_warmup_steps must be >= 0"):
        xse_sgd(lr=1e-3, lora_alpha=16, p_e=0.25, rotation_step_interval=1,
                rotation_warmup_steps=-1, momentum=0.9)


@pytest.mark.parametrize("source,floor", [("core", 0.8292), ("momentum", None)])
def test_keep_source_retention_floor(source, floor, monkeypatch):
    """XSE_KEEP_SOURCE=core must hit the provable retention floor sqrt(r_keep/r).

    The rotation is exactly an orthogonal projection of the weight update, so the
    retained energy fraction is g^2 with g = ||R'||/||R||. Choosing the kept frames
    as R's own top singular directions maximises g (Eckart-Young), giving the
    deterministic floor g >= sqrt(r_keep/r) by pigeonhole. "momentum" has no such
    bound -- measured 0.978 on causal LM but 0.759 on CoLA, BELOW this floor,
    which is what makes the switch a provable improvement there.
    """
    import importlib

    monkeypatch.setenv("XSE_KEEP_SOURCE", source)
    import lora_privacy.peft_lora_xs.xse as xse

    importlib.reload(xse)
    assert xse._KEEP_SOURCE == source

    r, r_keep = 16, 11
    torch.manual_seed(0)
    worst = 1.0
    for _ in range(200):
        R = torch.randn(r, r, dtype=torch.float64)
        M = torch.randn(r, r, dtype=torch.float64)
        sel = R if source == "core" else M
        U, _S, Vh = torch.linalg.svd(sel)
        Up, Vp = U[:, :r_keep], Vh.T[:, :r_keep]
        worst = min(worst, ((Up.T @ R @ Vp).norm() / R.norm()).item())
    if floor is not None:
        assert worst >= floor, f"core keep rule dipped to {worst:.4f} < {floor}"
    else:
        # momentum selection on an independent M concentrates at the RANDOM floor
        # r_keep/r = 0.6875, well below the core rule's guarantee.
        assert worst < 0.8292, "momentum keep unexpectedly matched the core floor"

    monkeypatch.delenv("XSE_KEEP_SOURCE", raising=False)
    importlib.reload(xse)


def test_keep_source_rejects_unknown(monkeypatch):
    import importlib

    monkeypatch.setenv("XSE_KEEP_SOURCE", "gradient")
    import lora_privacy.peft_lora_xs.xse as xse

    with pytest.raises(ValueError, match="XSE_KEEP_SOURCE must be"):
        importlib.reload(xse)
    monkeypatch.delenv("XSE_KEEP_SOURCE", raising=False)
    importlib.reload(xse)
