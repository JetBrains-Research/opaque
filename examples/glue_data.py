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

"""GLUE data + metrics for the sequence-classification path of train_causal_lm.py.

WHY THIS IS A SEPARATE MODULE AND NOT INLINE IN THE TRAINER. The trainer cannot
run on macOS (a pre-existing vmap/transformers incompatibility), so the only
local test available is an importable unit. Both LoRA-SB crashes came from
unit-testing against a hand-written stand-in for the real collate/loss instead of
the real thing, and each cost a GPU run to discover. Everything here is therefore
importable and CPU-testable, and the trainer calls exactly the functions the test
exercises.

The target is the LoRA-XS paper's GLUE setup (arXiv 2405.17604v3, ECAI 2025,
Table 1 + Appendix D.1) so our numbers can sit in their table: RoBERTa-large,
seq len 128, batch 32, warmup ratio 0.06, LoRA-XS on Wq/Wv/Wo/FC1, alpha 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from datasets import load_dataset


@dataclass(frozen=True)
class GlueTask:
    """One GLUE task's shape: which columns hold the text, and how it is scored."""

    name: str
    keys: tuple[str, ...]  # 1 for single-sentence, 2 for sentence-pair
    num_labels: int  # 1 means regression (STS-B)
    metric: str  # "accuracy" | "matthews" | "pearson"

    @property
    def is_regression(self) -> bool:
        return self.num_labels == 1


# The six tasks in the paper's Table 1, plus MNLI (needed because MRPC/RTE/STS-B
# are initialized from an MNLI-finetuned checkpoint there) and QQP for
# completeness. `metric` is the figure the paper reports for that column:
# Matthews for CoLA, Pearson for STS-B, accuracy elsewhere.
GLUE_TASKS: dict[str, GlueTask] = {
    "cola": GlueTask("cola", ("sentence",), 2, "matthews"),
    "sst2": GlueTask("sst2", ("sentence",), 2, "accuracy"),
    "mrpc": GlueTask("mrpc", ("sentence1", "sentence2"), 2, "accuracy"),
    "stsb": GlueTask("stsb", ("sentence1", "sentence2"), 1, "pearson"),
    "qnli": GlueTask("qnli", ("question", "sentence"), 2, "accuracy"),
    "rte": GlueTask("rte", ("sentence1", "sentence2"), 2, "accuracy"),
    "mnli": GlueTask("mnli", ("premise", "hypothesis"), 3, "accuracy"),
    "qqp": GlueTask("qqp", ("question1", "question2"), 2, "accuracy"),
}


def resolve_task(name: str) -> GlueTask:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in GLUE_TASKS:
        raise ValueError(
            f"Unknown GLUE task {name!r}. Known: {', '.join(sorted(GLUE_TASKS))}"
        )
    return GLUE_TASKS[key]


def build_glue_datasets(
    task: GlueTask,
    tokenizer,
    *,
    max_seq_len: int = 128,
    num_train_samples: int | None = None,
    num_eval_samples: int | None = None,
    seed: int = 42,
):
    """Return (train_dataset, eval_dataset), tokenized, columns removed.

    Eval is GLUE's own `validation` split, NOT a slice off the head of train.
    The causal-LM path takes the first `num_eval_samples` rows of the training
    stream as its eval set, which is fine for a language-modelling corpus but
    would be wrong here: GLUE's validation split is the set every published
    number is computed on, so anything else is not comparable to Table 1.

    MNLI's validation is split into matched/mismatched; we take matched, which is
    the column papers mean by "MNLI".
    """
    ds = load_dataset("glue", task.name)
    train = ds["train"]
    eval_split = "validation_matched" if task.name == "mnli" else "validation"
    evaluation = ds[eval_split]

    # Subsample only when asked. Full GLUE tasks are small (RTE 2.5k, MRPC 3.7k)
    # and the paper trains on all of them, so the default is everything.
    if num_train_samples is not None and num_train_samples < len(train):
        train = train.shuffle(seed=seed).select(range(num_train_samples))
    if num_eval_samples is not None and num_eval_samples < len(evaluation):
        evaluation = evaluation.select(range(num_eval_samples))

    def tokenize(batch):
        texts = [batch[k] for k in task.keys]
        return tokenizer(*texts, truncation=True, max_length=max_seq_len)

    # `label` is kept: the collate below needs it. Everything else goes, so the
    # collate never has to guess which columns are model inputs.
    keep = {"input_ids", "attention_mask", "label"}
    train = train.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in train.column_names if c not in keep],
        desc=f"Tokenizing {task.name} train",
    )
    evaluation = evaluation.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in evaluation.column_names if c not in keep],
        desc=f"Tokenizing {task.name} validation",
    )
    return train, evaluation


def make_glue_collate(task: GlueTask, tokenizer, device):
    """Build a collate returning ``(input_ids, attention_mask, labels)``.

    A 3-tuple, deliberately. The causal path's collate returns a 1-tuple and its
    grad functions are built with ``batch_argnums=(1,)``; classification needs the
    attention mask (RoBERTa has no causal mask to fall back on, so padding would
    otherwise be attended) and a separate label, hence ``batch_argnums=(1, 2, 3)``.
    Multi-element batch_argnums is an already-supported path -- see
    packages/opaque-core/tests/clipping/test_empty_batch.py.

    Padding is done here rather than by DataCollatorWithPadding so the returned
    object is a plain tuple of device tensors, which is what the vmap'd
    per-example grad call sites expect.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:  # RoBERTa always has one; guard anyway
        raise ValueError("tokenizer has no pad_token_id; cannot pad a GLUE batch")
    label_dtype = torch.float32 if task.is_regression else torch.long

    def collate(examples):
        width = max(len(ex["input_ids"]) for ex in examples)
        ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(examples), width), dtype=torch.long)
        for i, ex in enumerate(examples):
            n = len(ex["input_ids"])
            ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            # Prefer the tokenizer's own mask; fall back to "non-pad" only if the
            # column is absent, since a real token can equal pad_id in principle.
            am = ex.get("attention_mask")
            mask[i, :n] = (
                torch.tensor(am, dtype=torch.long) if am is not None else 1
            )
        labels = torch.tensor([ex["label"] for ex in examples], dtype=label_dtype)
        return (ids.to(device), mask.to(device), labels.to(device))

    return collate


# ---------------------------------------------------------------- metrics


def _matthews(preds: list[int], refs: list[int]) -> float:
    """Matthews correlation for the binary case (CoLA's reported figure).

    Written out rather than pulled from sklearn: sklearn is not in the training
    image, and adding a dependency to compute a four-term ratio is not a good
    trade. Returns 0.0 when a denominator term vanishes, which is sklearn's
    documented convention for a degenerate confusion matrix.
    """
    tp = sum(1 for p, r in zip(preds, refs) if p == 1 and r == 1)
    tn = sum(1 for p, r in zip(preds, refs) if p == 0 and r == 0)
    fp = sum(1 for p, r in zip(preds, refs) if p == 1 and r == 0)
    fn = sum(1 for p, r in zip(preds, refs) if p == 0 and r == 1)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _pearson(preds: list[float], refs: list[float]) -> float:
    """Pearson correlation (STS-B's reported figure)."""
    n = len(preds)
    if n < 2:
        return 0.0
    mp = sum(preds) / n
    mr = sum(refs) / n
    cov = sum((p - mp) * (r - mr) for p, r in zip(preds, refs))
    vp = sum((p - mp) ** 2 for p in preds)
    vr = sum((r - mr) ** 2 for r in refs)
    if vp <= 0.0 or vr <= 0.0:
        return 0.0
    return cov / math.sqrt(vp * vr)


def glue_metric(task: GlueTask, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Score one full validation pass. Returns the figure the paper reports.

    `logits` is (N, num_labels) -- or (N, 1)/(N,) for STS-B. Higher is better for
    all three metrics, so callers can treat it uniformly as a score.
    """
    if task.is_regression:
        preds = logits.detach().float().reshape(-1).cpu().tolist()
        refs = labels.detach().float().reshape(-1).cpu().tolist()
        return _pearson(preds, refs)

    preds = logits.detach().float().argmax(dim=-1).cpu().tolist()
    refs = labels.detach().long().reshape(-1).cpu().tolist()
    if task.metric == "matthews":
        return _matthews(preds, refs)
    return sum(1 for p, r in zip(preds, refs) if p == r) / max(len(refs), 1)
