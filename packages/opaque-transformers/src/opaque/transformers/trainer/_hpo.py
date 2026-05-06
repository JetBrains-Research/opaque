"""Hyperparameter-search helpers for DPTrainer.

The public surface mirrors HuggingFace's ``Trainer.hyperparameter_search`` but
keeps execution local to this trainer.  HF's Optuna runner assumes Accelerate
objects (``accelerator``, ``model_wrapped``) that DPTrainer deliberately does
not own, so the supported path is implemented here instead of delegated.
"""

from __future__ import annotations

import gc
import importlib
import logging
from collections.abc import Mapping
from typing import Any, Callable

import torch
from transformers.trainer_utils import (
    BestRun,
    HPSearchBackend,
    default_compute_objective,
    default_hp_space_optuna,
    default_hp_space_wandb,
)

__all__ = ["hyperparameter_search", "is_multi_objective_study"]

log = logging.getLogger(__name__)


def hyperparameter_search(
    trainer: Any,
    hp_space: Callable[[Any], dict[str, float]] | None = None,
    compute_objective: Callable[[dict[str, float]], float] | None = None,
    n_trials: int = 20,
    direction: str | list[str] = "minimize",
    backend: str | HPSearchBackend | None = None,
    hp_name: Callable[[Any], str] | None = None,
    **kwargs: Any,
) -> BestRun | list[BestRun]:
    """Run local HPO and return HF-compatible ``BestRun`` objects."""
    selected = HPSearchBackend(backend or HPSearchBackend.OPTUNA)
    if selected not in {HPSearchBackend.OPTUNA, HPSearchBackend.WANDB}:
        backend_name = selected.value
        if selected == HPSearchBackend.RAY:
            raise ValueError(
                "DPTrainer.hyperparameter_search(backend='ray') is not a local "
                "trial backend. Ray Tune owns process/actor execution, checkpoint "
                "marshalling, and distributed trial resources; DPTrainer only "
                "supports local backend='optuna' and backend='wandb' sweeps until "
                "a DP-aware external execution layer is implemented."
            )
        raise ValueError(
            "DPTrainer.hyperparameter_search currently supports local "
            "backend='optuna' and backend='wandb' sweeps only; "
            f"got backend={backend_name!r}."
        )
    if trainer.model_init is None:
        raise RuntimeError(
            "To use hyperparameter search, pass a `model_init` function to DPTrainer "
            "so each trial starts from a freshly initialized model."
        )

    if selected == HPSearchBackend.WANDB:
        return _run_wandb_search(
            trainer,
            hp_space=hp_space,
            compute_objective=compute_objective,
            n_trials=n_trials,
            direction=direction,
            hp_name=hp_name,
            **kwargs,
        )
    return _run_optuna_search(
        trainer,
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=n_trials,
        direction=direction,
        hp_name=hp_name,
        **kwargs,
    )


def _run_optuna_search(
    trainer: Any,
    hp_space: Callable[[Any], dict[str, float]] | None,
    compute_objective: Callable[[dict[str, float]], float] | None,
    n_trials: int,
    direction: str | list[str],
    hp_name: Callable[[Any], str] | None,
    **kwargs: Any,
) -> BestRun | list[BestRun]:
    """Run local Optuna HPO."""

    try:
        optuna = importlib.import_module("optuna")
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "DPTrainer.hyperparameter_search(backend='optuna') requires optuna; "
            "install it with `pip install optuna`."
        ) from exc

    trainer.hp_search_backend = HPSearchBackend.OPTUNA
    trainer.hp_space = hp_space or default_hp_space_optuna
    trainer.hp_name = hp_name
    trainer.compute_objective = compute_objective or default_compute_objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    if n_jobs != 1:
        raise ValueError(
            "DPTrainer.hyperparameter_search uses one stateful trainer instance; "
            "Optuna n_jobs must be 1. Launch independent DPTrainer processes for "
            "parallel sweeps."
        )
    gc_after_trial = kwargs.pop("gc_after_trial", False)
    directions = direction if isinstance(direction, list) else None
    study_direction = None if directions is not None else direction
    study = optuna.create_study(
        direction=study_direction,
        directions=directions,
        **kwargs,
    )

    def objective(trial: Any) -> float | list[float]:
        trainer.objective = None
        trainer.train(resume_from_checkpoint=None, trial=trial)
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        _release_memory(trainer)
        return trainer.objective

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            gc_after_trial=gc_after_trial,
        )
        if is_multi_objective_study(study):
            return [
                BestRun(str(trial.number), trial.values, trial.params)
                for trial in study.best_trials
            ]
        best = study.best_trial
        return BestRun(str(best.number), best.value, best.params)
    finally:
        _clear_hpo_state(trainer)


def _run_wandb_search(
    trainer: Any,
    hp_space: Callable[[Any], dict[str, Any]] | None,
    compute_objective: Callable[[dict[str, float]], float] | None,
    n_trials: int,
    direction: str | list[str],
    hp_name: Callable[[Any], str] | None,
    **kwargs: Any,
) -> BestRun:
    """Run a local W&B sweep via ``wandb.agent``."""
    if isinstance(direction, list):
        raise NotImplementedError("W&B sweeps support a single objective direction.")
    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "W&B sweeps require direction='minimize' or direction='maximize'."
        )
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:  # pragma: no cover - exercised only without wandb
        raise ImportError(
            "DPTrainer.hyperparameter_search(backend='wandb') requires wandb; "
            "install it with `pip install wandb`."
        ) from exc

    trainer.hp_search_backend = HPSearchBackend.WANDB
    trainer.hp_space = hp_space or default_hp_space_wandb
    trainer.hp_name = hp_name
    trainer.compute_objective = compute_objective or default_compute_objective
    _maybe_add_wandb_callback(trainer)

    best_trial: dict[str, Any] = {
        "run_id": None,
        "objective": None,
        "hyperparameters": None,
    }
    sweep_id = kwargs.pop("sweep_id", None)
    project = kwargs.pop("project", None)
    name = kwargs.pop("name", None)
    entity = kwargs.pop("entity", None)
    metric = kwargs.pop("metric", "eval/loss")
    if kwargs:
        raise TypeError(
            "Unsupported W&B hyperparameter_search kwargs: "
            f"{', '.join(sorted(kwargs))}."
        )

    sweep_config = dict(trainer.hp_space(None))
    sweep_metric = dict(sweep_config.get("metric", {}))
    sweep_metric["goal"] = direction
    sweep_metric["name"] = metric
    sweep_config["metric"] = sweep_metric
    if name:
        sweep_config["name"] = name

    def objective() -> Any:
        run = getattr(wandb, "run", None) or wandb.init()
        trainer.state.trial_name = getattr(run, "name", None)
        run_config = getattr(run, "config", None)
        if hasattr(run_config, "update"):
            run_config.update({"assignments": {}, "metric": metric})
        config = getattr(wandb, "config", run_config)
        trial_params = _wandb_config_items(config)
        run_id = getattr(run, "id", None) or getattr(run, "name", None)
        if run_id is not None:
            trial_params.setdefault("run_id", run_id)
        trial_params["wandb"] = True

        trainer.objective = None
        trainer.train(resume_from_checkpoint=None, trial=trial_params)
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)

        current_objective = trainer.objective
        is_better = best_trial["run_id"] is None or (
            current_objective < best_trial["objective"]
            if direction == "minimize"
            else current_objective > best_trial["objective"]
        )
        if is_better:
            best_trial["run_id"] = run_id
            best_trial["objective"] = current_objective
            best_trial["hyperparameters"] = dict(trial_params)
        _release_memory(trainer)
        return current_objective

    try:
        if not sweep_id:
            sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        else:
            wandb_env = getattr(wandb, "env", None)
            if wandb_env is not None:
                if entity and hasattr(wandb_env, "set_entity"):
                    wandb_env.set_entity(entity)
                if project and hasattr(wandb_env, "set_project"):
                    wandb_env.set_project(project)
        log.info("wandb sweep id - %s", sweep_id)
        wandb.agent(sweep_id, function=objective, count=n_trials)
        return BestRun(
            best_trial["run_id"],
            best_trial["objective"],
            best_trial["hyperparameters"] or {},
            sweep_id,
        )
    finally:
        _clear_hpo_state(trainer)


def is_multi_objective_study(study: Any) -> bool:
    """Return whether an Optuna study has multiple objectives."""
    checker = getattr(study, "_is_multi_objective", None)
    if callable(checker):
        return bool(checker())
    directions = getattr(study, "directions", None)
    return directions is not None and len(directions) > 1


def _wandb_config_items(config: Any) -> dict[str, Any]:
    """Extract plain trial parameters from W&B's config object."""
    if config is None:
        return {}
    items = getattr(config, "_items", None)
    if isinstance(items, Mapping):
        return dict(items)
    if isinstance(config, Mapping):
        return dict(config)
    try:
        return dict(config)
    except (TypeError, ValueError):
        pass
    values = vars(config).get("_items", vars(config))
    return dict(values) if isinstance(values, Mapping) else {}


def _maybe_add_wandb_callback(trainer: Any) -> None:
    """Register W&B reporting callback when Transformers can construct it."""
    try:
        from transformers.integrations import WandbCallback
    except Exception:  # pragma: no cover - best-effort integration guard
        return

    from ._callback import wrap_reporting_callback_class

    callbacks = getattr(getattr(trainer, "callback_handler", None), "callbacks", [])
    if any(isinstance(callback, WandbCallback) for callback in callbacks):
        return
    try:
        trainer.add_callback(wrap_reporting_callback_class(WandbCallback)())
    except Exception as exc:  # pragma: no cover - optional integration guard
        log.debug("Could not register WandbCallback for HPO sweep: %s", exc)


def _clear_hpo_state(trainer: Any) -> None:
    trainer.hp_search_backend = None
    trainer.hp_space = None
    trainer.hp_name = None
    trainer.compute_objective = None


def _release_memory(trainer: Any) -> None:
    """Best-effort memory cleanup between local trials."""
    gc.collect()
    device = getattr(trainer, "_device", None)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    elif getattr(device, "type", None) == "mps":
        torch.mps.empty_cache()
