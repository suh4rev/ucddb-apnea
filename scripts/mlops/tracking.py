from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_EXPERIMENT_NAME = "ucddb-apnea"
DEFAULT_WANDB_PROJECT = "ucddb-apnea"


@dataclass
class Tracker:
    mlflow: Any = None
    wandb_run: Any = None


def is_mlflow_enabled() -> bool:
    value = os.getenv("UCDDB_ENABLE_MLFLOW") or os.getenv("UCDBB_ENABLE_MLFLOW")
    return str(value).strip().lower() in TRUE_VALUES


def is_wandb_enabled() -> bool:
    value = os.getenv("UCDDB_ENABLE_WANDB")
    return str(value).strip().lower() in TRUE_VALUES


def _as_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _load_mlflow():
    if not is_mlflow_enabled():
        return None

    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        print(
            "MLflow tracking requested but mlflow is not installed. "
            "Install optional dependencies with: pip install -r requirements-mlops.txt"
        )
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def _start_wandb(run_name: str, tags: dict[str, Any] | None = None):
    if not is_wandb_enabled():
        return None

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        print(
            "W&B tracking requested but wandb is not installed. "
            "Install optional dependencies with: pip install -r requirements-mlops.txt"
        )
        return None

    project = os.getenv("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    wandb_tags = []
    if tags:
        wandb_tags = [f"{key}:{value}" for key, value in tags.items() if value is not None]

    return wandb.init(
        project=project,
        name=run_name,
        tags=wandb_tags,
        reinit=True,
    )


def _get_mlflow(tracker: Any) -> Any:
    return tracker.mlflow if isinstance(tracker, Tracker) else tracker


def _get_wandb_run(tracker: Any) -> Any:
    return tracker.wandb_run if isinstance(tracker, Tracker) else None


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[Any]:
    mlflow = _load_mlflow()
    wandb_run = _start_wandb(run_name, tags)

    if mlflow is None and wandb_run is None:
        yield None
        return

    tracker = Tracker(mlflow=mlflow, wandb_run=wandb_run)
    try:
        if mlflow is None:
            yield tracker
            return

        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(
                    {key: str(value) for key, value in tags.items() if value is not None}
                )
            yield tracker
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def log_params(mlflow: Any, params: dict[str, Any]) -> None:
    if mlflow is None:
        return

    clean_params = {
        key: _as_scalar(value)
        for key, value in params.items()
        if _as_scalar(value) is not None
    }
    if not clean_params:
        return

    mlflow_client = _get_mlflow(mlflow)
    if mlflow_client is not None:
        mlflow_client.log_params(clean_params)

    wandb_run = _get_wandb_run(mlflow)
    if wandb_run is not None:
        wandb_run.config.update(clean_params, allow_val_change=True)


def log_metrics(mlflow: Any, metrics: dict[str, Any]) -> None:
    if mlflow is None:
        return

    clean_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        value = _as_scalar(value)
        if value is None:
            continue
        clean_metrics[key] = float(value)

    if not clean_metrics:
        return

    mlflow_client = _get_mlflow(mlflow)
    if mlflow_client is not None:
        for key, value in clean_metrics.items():
            mlflow_client.log_metric(key, value)

    wandb_run = _get_wandb_run(mlflow)
    if wandb_run is not None:
        wandb_run.log(clean_metrics)


def log_artifacts(mlflow: Any, paths: list[Path], artifact_path: str = "artifacts") -> None:
    if mlflow is None:
        return

    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return

    mlflow_client = _get_mlflow(mlflow)
    if mlflow_client is not None:
        for path in existing_paths:
            mlflow_client.log_artifact(str(path), artifact_path=artifact_path)

    wandb_run = _get_wandb_run(mlflow)
    if wandb_run is None:
        return

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        return

    artifact = wandb.Artifact(artifact_path, type="analysis")
    for path in paths:
        if path.exists():
            artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)
