"""ZenML pipeline for the Rényi effective-rank DP experiments.

Wraps ``examples/train_causal_lm.py`` (see ``docs/renyi-dp-experiment-handoff.md``)
as a single GPU step so the two-arm experiment runs on the non-TRACE Kubernetes
GPU stack instead of Cadence. The step shells out to the trainer CLI verbatim —
the same command the handoff doc and ``.cadence/configs/renyi_dp_vs_nodp.yaml``
run — so behaviour is identical to a local/Cadence launch.

This module is what ZenML executes *inside the pod*, so it deliberately depends
only on ``zenml`` + the stdlib. All infrastructure (image, GPU, resources,
secret env vars) is attached at submit time in ``run.py`` via ``settings.py``
(which uses ``jb-mlops``), mirroring how NES builds its step settings.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from zenml import pipeline, step

LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Locate the opaque repo root at runtime.

    Prefers an explicit ``OPAQUE_REPO_DIR`` override, then the copy baked into
    the training image at ``/opt/opaque``, then a walk up from this file (local
    ``python deploy/zenml/pipeline.py`` style runs).
    """
    override = os.environ.get("OPAQUE_REPO_DIR")
    if override and (Path(override) / "examples" / "train_causal_lm.py").exists():
        return Path(override)

    baked = Path("/opt/opaque")
    if (baked / "examples" / "train_causal_lm.py").exists():
        return baked

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "examples" / "train_causal_lm.py").exists():
            return candidate

    return baked


@step(enable_cache=False)
def train_arm(
    cmd_args: list[str],
    run_name: str,
    scratch_dir: str = "/scratch",
) -> dict[str, Any]:
    """Run one training arm by invoking the trainer CLI in a subprocess.

    W&B / HF credentials and ``WANDB_*`` targets arrive as pod env vars (injected
    from the ``ai-for-code`` ZenML secret by the settings in ``run.py``); this
    step only pins ``RUN_NAME`` and routes caches onto the scratch volume.
    """
    root = _repo_root()
    script = root / "examples" / "train_causal_lm.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find examples/train_causal_lm.py under {root}. "
            "Set OPAQUE_REPO_DIR or bake the repo into the image at /opt/opaque."
        )

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["RUN_NAME"] = run_name

    # Route large caches / temp / HF downloads onto the mounted scratch volume so
    # they don't exhaust the container root filesystem.
    scratch = Path(scratch_dir)
    caches = {
        "HF_HOME": scratch / "hf",
        "HF_HUB_CACHE": scratch / "hf" / "hub",
        "TMPDIR": scratch / "tmp",
        "WANDB_DIR": scratch / "wandb",
    }
    for key, path in caches.items():
        env.setdefault(key, str(path))
        try:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.warning("Could not create %s=%s (%s)", key, env[key], exc)

    # ZenML injects secret values verbatim, and the shared ``ai-for-code`` secret
    # can carry stray bytes around the token (a trailing newline from the secret
    # store, or an invisible/zero-width char from a copy-paste). wandb 0.28's key
    # validator (wbauth) accepts only ``[\w-]`` in the API key and rejects
    # anything else with "API key may only contain the letters A-Z, digits and
    # underscores", so a single stray byte fails ``wandb.init()``. ``str.strip()``
    # only catches leading/trailing standard whitespace, so instead keep exactly
    # the characters the validator allows — a valid W&B/HF token (including an
    # on-prem ``local-`` prefix) is entirely ``[\w-]`` anyway. Log the dropped
    # code points (not the secret) so a genuinely malformed token is diagnosable.
    for _cred in ("WANDB_API_KEY", "HF_TOKEN"):
        raw = env.get(_cred)
        if not raw:
            continue
        cleaned = re.sub(r"[^\w-]", "", raw)
        if cleaned != raw:
            dropped = [f"U+{ord(c):04X}" for c in raw if not re.fullmatch(r"[\w-]", c)]
            LOGGER.warning(
                "%s carried %d non-token char(s) %s; sanitized (len %d -> %d)",
                _cred,
                len(raw) - len(cleaned),
                dropped,
                len(raw),
                len(cleaned),
            )
        env[_cred] = cleaned

    if env.get("WANDB_API_KEY"):
        env.setdefault("WANDB_MODE", "online")

    argv = [sys.executable, str(script), *cmd_args]
    LOGGER.info("cwd=%s", root)
    LOGGER.info("exec: %s", " ".join(shlex.quote(a) for a in argv))

    proc = subprocess.run(argv, cwd=str(root), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"training arm {run_name!r} failed (exit code {proc.returncode})")

    return {"run_name": run_name, "returncode": proc.returncode, "argv": argv}


@pipeline(enable_cache=False, name="Opaque/RenyiDP/Train")
def renyi_training_pipeline(
    cmd_args: list[str],
    run_name: str,
    scratch_dir: str = "/scratch",
) -> None:
    """One pipeline run == one experiment arm (DP, non-DP, or smoke)."""
    train_arm(cmd_args=cmd_args, run_name=run_name, scratch_dir=scratch_dir)
