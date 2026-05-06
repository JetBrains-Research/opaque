"""Hyperparameter-search helpers for DPTrainer.

The public surface mirrors HuggingFace's ``Trainer.hyperparameter_search``.
Optuna and W&B sweep agents run locally with one stateful trainer reused
across trials.  Ray Tune is dispatched out-of-process via Ray's actor
model: a single trainer instance is pickled by ``tune.with_parameters``
and shipped to each trial actor, which then calls
``trainer.train(trial=config_dict)``.  Multi-rank Ray trials (per-trial
DDP) are gated until Phase 10 lands.
"""

from __future__ import annotations

import functools
import gc
import importlib
import importlib.util
import logging
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import (
    BestRun,
    HPSearchBackend,
    IntervalStrategy,
    TrainerMemoryTracker,
    default_compute_objective,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_wandb,
)

from opaque.transformers.trainer import _checkpoint as ckpt

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
    """Run HPO and return HF-compatible ``BestRun`` objects."""
    selected = HPSearchBackend(backend or HPSearchBackend.OPTUNA)
    if selected not in {
        HPSearchBackend.OPTUNA,
        HPSearchBackend.WANDB,
        HPSearchBackend.RAY,
    }:
        raise ValueError(
            "DPTrainer.hyperparameter_search supports backend in "
            "{'optuna', 'wandb', 'ray'}; "
            f"got backend={selected.value!r}."
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
    if selected == HPSearchBackend.RAY:
        return _run_ray_search(
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
            "install it with `pip install opaque-transformers[optuna-hpo]` "
            "(or `pip install optuna`)."
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
            "install it with `pip install opaque-transformers[wandb-hpo]` "
            "(or `pip install wandb`)."
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


# ---------------------------------------------------------------------
# Ray Tune backend
# ---------------------------------------------------------------------
#
# Layout mirrors HF's ``transformers.integrations.run_hp_search_ray``
# (transformers 4.57): one trainer is pickled by
# ``ray.tune.with_parameters`` and shipped to each Tune actor.  The
# actor calls ``trainer.train(trial=config_dict)``, which lets the
# trainer reuse its existing dict-trial path (model_init reinvocation,
# trial-scoped output dir, callback handler rebuild).  Ray's per-trial
# checkpoint is written to a temp dir via
# ``DPTrainer._tune_save_checkpoint`` and reported via
# ``ray.train.report(metrics, checkpoint=...)``.


def _run_ray_search(
    trainer: Any,
    hp_space: Callable[[Any], dict[str, Any]] | None,
    compute_objective: Callable[[dict[str, float]], float] | None,
    n_trials: int,
    direction: str | list[str],
    hp_name: Callable[[Any], str] | None,
    **kwargs: Any,
) -> BestRun:
    """Run a Ray Tune sweep and return an HF-compatible ``BestRun``."""
    try:
        ray = importlib.import_module("ray")
        importlib.import_module("ray.train")
        importlib.import_module("ray.tune")
    except ImportError as exc:  # pragma: no cover - exercised only without ray
        raise ImportError(
            "DPTrainer.hyperparameter_search(backend='ray') requires Ray Tune; "
            "install it with `pip install opaque-transformers[ray-hpo]` "
            "(or `pip install 'ray[tune]'`)."
        ) from exc

    if isinstance(direction, list):
        raise NotImplementedError("Ray Tune supports a single objective direction.")
    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "Ray HPO requires direction='minimize' or direction='maximize'; "
            f"got {direction!r}."
        )

    trainer.hp_search_backend = HPSearchBackend.RAY
    trainer.hp_space = hp_space or default_hp_space_ray
    trainer.hp_name = hp_name
    trainer.compute_objective = compute_objective or default_compute_objective

    _reject_multirank_for_ray(trainer, kwargs)
    _scrub_for_pickling(trainer)

    def _objective(trial: dict[str, Any], local_trainer: Any) -> None:
        try:
            from transformers.utils.notebook import NotebookProgressCallback

            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:  # pragma: no cover - notebook absent
            pass

        local_trainer.objective = None
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            # HF parity workaround: reset of ``objective`` to None on
            # resume can drive an unnecessary final checkpoint when
            # ``train`` is a no-op.
            local_trainer.objective = "objective"
            with checkpoint.as_directory() as checkpoint_dir:
                resume_path = next(
                    Path(checkpoint_dir).glob(f"{ckpt.PREFIX_CHECKPOINT_DIR}-*")
                ).as_posix()
                local_trainer.train(resume_from_checkpoint=resume_path, trial=trial)
        else:
            local_trainer.train(trial=trial)

        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            metrics["objective"] = local_trainer.objective
            metrics["done"] = True
            with tempfile.TemporaryDirectory() as temp_dir:
                local_trainer._tune_save_checkpoint(checkpoint_dir=temp_dir)
                report_checkpoint = ray.train.Checkpoint.from_directory(temp_dir)
                ray.train.report(metrics, checkpoint=report_checkpoint)

    _tb_writer = _pop_tensorboard_callback(trainer)
    _set_default_resources(trainer, kwargs)
    _set_default_progress_reporter(kwargs)
    _validate_scheduler_requires_eval(trainer, kwargs)

    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)
    trainable = _wrap_dynamic_modules(trainable)

    try:
        analysis = ray.tune.run(
            trainable,
            config=trainer.hp_space(None),
            num_samples=n_trials,
            **kwargs,
        )
        best_trial = analysis.get_best_trial(
            metric="objective",
            mode=direction[:3],
            scope=trainer.args.ray_scope,
        )
        best_run = BestRun(
            best_trial.trial_id,
            best_trial.last_result["objective"],
            best_trial.config,
            analysis,
        )
        return best_run
    finally:
        if _tb_writer is not None:
            try:
                trainer.add_callback(_tb_writer)
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("Could not restore TensorBoardCallback: %s", exc)
        _clear_hpo_state(trainer)


def _reject_multirank_for_ray(trainer: Any, kwargs: dict[str, Any]) -> None:
    """Block multi-rank trainer and per-trial DDP — Phase 10 prerequisite.

    DPTrainer's distributed semantics (``parallel_poisson`` accounting,
    ``local_shard`` rank policy, gradient synchronization) land in Phase
    10.  Until then, refuse Ray sweeps where any trial would run more
    than one rank.
    """
    world_size = getattr(getattr(trainer, "args", None), "world_size", 1) or 1
    if world_size > 1:
        raise ValueError(
            "DPTrainer.hyperparameter_search(backend='ray') does not yet support "
            "multi-rank trainer processes (args.world_size > 1).  This requires "
            "Phase 10 (DDP — local_shard / parallel_poisson sampling, "
            "rank-aware accountants, gradient gather) to land first."
        )
    resources = kwargs.get("resources_per_trial") or {}
    gpus_per_trial = resources.get("gpu", 0) if isinstance(resources, Mapping) else 0
    if gpus_per_trial and gpus_per_trial > 1:
        raise ValueError(
            "DPTrainer.hyperparameter_search(backend='ray') does not yet support "
            "per-trial DDP (resources_per_trial['gpu'] > 1).  Phase 10 is the "
            "prerequisite (local_shard + parallel_poisson + distributed eval)."
        )


def _scrub_for_pickling(trainer: Any) -> None:
    """Make a DPTrainer picklable for ``tune.with_parameters``.

    Mirrors HF's pre-launch scrubs: drop the live model (the actor
    rebuilds it via ``call_model_init(trial)``), swap the memory tracker
    to a skip-only instance (its ``psutil`` thread is not picklable),
    and clear cached dataloader handles.  Also asserts ``output_dir`` is
    absolute — Ray Tune chdirs each actor and a relative path would
    silently land checkpoints under Tune's per-trial working directory.
    """
    args = getattr(trainer, "args", None)
    output_dir = getattr(args, "output_dir", None) if args is not None else None
    if output_dir is not None and not os.path.isabs(output_dir):
        raise ValueError(
            "Ray Tune actors run with their own working directory; "
            f"args.output_dir={output_dir!r} must be an absolute path."
        )

    skip = (
        bool(getattr(args, "skip_memory_metrics", True)) if args is not None else True
    )
    trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=skip or True)
    # Drop the live model — each actor rebuilds it via call_model_init(trial).
    trainer.model = None
    # Cached dataloaders hold worker handles that don't pickle.
    if hasattr(trainer, "_train_dataloader"):
        trainer._train_dataloader = None
    if hasattr(trainer, "_eval_dataloader"):
        trainer._eval_dataloader = None


def _pop_tensorboard_callback(trainer: Any) -> Any | None:
    """Pop the TensorBoard callback before launching Ray (HF parity).

    Tensorboard's ``SummaryWriter`` holds a thread + open file and does
    not pickle.  HF strips it before ``tune.with_parameters`` and re-
    attaches after ``tune.run`` returns.
    """
    try:
        from transformers.integrations import TensorBoardCallback
    except ImportError:  # pragma: no cover - tensorboard missing
        return None
    return trainer.pop_callback(TensorBoardCallback)


def _set_default_resources(trainer: Any, kwargs: dict[str, Any]) -> None:
    """HF parity: default to {'cpu': 1, 'gpu': 1 if cuda else 0}."""
    if "resources_per_trial" in kwargs:
        return
    resources: dict[str, int] = {"cpu": 1}
    has_cuda = (
        hasattr(torch, "cuda")
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    )
    if has_cuda:
        resources["gpu"] = 1
    kwargs["resources_per_trial"] = resources
    log.info(
        "No `resources_per_trial` arg was passed into "
        "`hyperparameter_search`. Setting it to a default value of %s.",
        resources,
    )


def _set_default_progress_reporter(kwargs: dict[str, Any]) -> None:
    """HF parity: default to ``CLIReporter(metric_columns=['objective'])``."""
    if "progress_reporter" in kwargs:
        return
    try:
        from ray.tune import CLIReporter
    except ImportError:  # pragma: no cover - ray.tune absent
        return
    kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])


def _validate_scheduler_requires_eval(trainer: Any, kwargs: dict[str, Any]) -> None:
    """HF parity: ASHA / Hyperband / Median / PBT schedulers need eval steps.

    Schedulers that prune trials early need intermediate metric reports;
    those reports come from ``_report_to_hp_search`` which fires only
    when ``eval_strategy != NO``.  Mirror HF's pre-flight check.
    """
    scheduler = kwargs.get("scheduler")
    if scheduler is None:
        return
    try:
        from ray.tune.schedulers import (
            ASHAScheduler,
            HyperBandForBOHB,
            MedianStoppingRule,
            PopulationBasedTraining,
        )
    except ImportError:  # pragma: no cover - ray.tune absent
        return
    intermediate_classes = (
        ASHAScheduler,
        MedianStoppingRule,
        HyperBandForBOHB,
        PopulationBasedTraining,
    )
    if not isinstance(scheduler, intermediate_classes):
        return
    args = trainer.args
    eval_off = (
        not getattr(args, "do_eval", False) or args.eval_strategy == IntervalStrategy.NO
    )
    if eval_off:
        cls_name = type(scheduler).__name__
        raise RuntimeError(
            f"You are using {cls_name} as a scheduler but you haven't enabled "
            "evaluation during training. This means your trials will not "
            "report intermediate results to Ray Tune, and can thus not be "
            "stopped early or used to exploit other trials parameters. "
            f"If this is what you want, do not use {cls_name}. If you would "
            f"like to use {cls_name}, make sure you pass `do_eval=True` and "
            "`eval_strategy='steps'` in the Trainer `args`."
        )


def _wrap_dynamic_modules(trainable: Any) -> Any:
    """Ensure ``datasets_modules`` is loaded inside each Tune actor.

    HF parity: see https://github.com/huggingface/transformers/issues/11565.
    Without this wrapper, actors that need ``datasets`` dynamic modules
    (custom dataset loaders) raise ``ImportError`` because Tune's worker
    process does not inherit ``sys.modules`` from the driver.
    """

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args: Any, **kwargs: Any) -> Any:
        try:
            datasets_load = importlib.import_module("datasets.load")
        except ImportError:  # pragma: no cover - datasets absent
            return trainable(*args, **kwargs)
        try:
            dynamic_modules_path = os.path.join(
                datasets_load.init_dynamic_modules(), "__init__.py"
            )
            spec = importlib.util.spec_from_file_location(
                "datasets_modules", dynamic_modules_path
            )
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        except Exception as exc:  # pragma: no cover - best-effort
            log.debug("Skipping datasets dynamic-modules preload in Ray actor: %s", exc)
        return trainable(*args, **kwargs)

    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__
    return dynamic_modules_import_trainable
