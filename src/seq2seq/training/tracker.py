"""
MLflow Experiment Tracking Module for Tiny-Seq2Seq.
Provides standardized, production-grade telemetry, parameter logging,
metric recording, artifact management, and nested ablation study tracking.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import mlflow
from mlflow.entities import Run

from utils.logger import logger


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens nested dictionaries for MLflow parameter logging."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


class MLflowTracker:
    """
    Manages experiment tracking and artifact logging via MLflow.
    Designed for resilience: logging failures emit warnings without halting training.
    """

    def __init__(
        self,
        experiment_name: str = "TinySeq2Seq-Training",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        self.artifact_location = artifact_location

        # Set MLflow backend
        mlflow.set_tracking_uri(self.tracking_uri)

        # Get or create experiment
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location,
            )
        else:
            self.experiment_id = self.experiment.experiment_id

        self.active_run: Optional[Run] = None

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Iterator[Run]:
        """Context manager to start and automatically close an MLflow run."""
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description,
            )
            self.active_run = run
            logger.info(
                f"Started MLflow run '{run_name or run.info.run_id}' "
                f"(ID: {run.info.run_id}, Experiment: {self.experiment_name})"
            )
            yield run
        except Exception as e:
            logger.error(f"Error during MLflow run execution: {e}")
            raise
        finally:
            if mlflow.active_run() is not None:
                mlflow.end_run()
                self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Flattens and logs hyperparameter dictionary to active run."""
        try:
            if mlflow.active_run() is None:
                logger.warning("Attempted to log params with no active MLflow run.")
                return
            flattened = _flatten_dict(params)
            mlflow.log_params(flattened)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_step_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Logs real-time training step metrics with a step index."""
        try:
            if mlflow.active_run() is None:
                return
            cleaned_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if v is not None and not (isinstance(v, float) and (v != v))  # Filter NaN
            }
            mlflow.log_metrics(cleaned_metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log step metrics to MLflow at step {step}: {e}")

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logs validation and aggregate epoch metrics with an epoch index."""
        try:
            if mlflow.active_run() is None:
                return
            cleaned_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if v is not None and not (isinstance(v, float) and (v != v))
            }
            mlflow.log_metrics(cleaned_metrics, step=epoch)
        except Exception as e:
            logger.warning(f"Failed to log epoch metrics to MLflow at epoch {epoch}: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Uploads a local file or directory as an artifact of the active run."""
        try:
            if mlflow.active_run() is None:
                logger.warning(f"No active run to log artifact: {local_path}")
                return
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            else:
                logger.warning(f"Artifact file not found: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def log_figure(self, fig: Any, artifact_file: str) -> None:
        """Logs a matplotlib or plotly figure to MLflow."""
        try:
            if mlflow.active_run() is None:
                return
            mlflow.log_figure(fig, artifact_file)
        except Exception as e:
            logger.warning(f"Failed to log figure {artifact_file}: {e}")

    def log_translation_samples(
        self,
        samples: List[Dict[str, Any]],
        epoch: int,
        artifact_path: str = "translations",
    ) -> None:
        """
        Formats sample translations into a Markdown table artifact and logs it to MLflow.
        Each sample dict should have: 'source', 'target', 'prediction', and optional 'beam_size'.
        """
        try:
            if mlflow.active_run() is None or not samples:
                return

            md_lines = [
                f"# Canary Sample Translations - Epoch {epoch}",
                "",
                "| Index | Source (Input) | Reference (Target) | Model Prediction | Beam |",
                "|:---:|:---|:---|:---|:---:|",
            ]

            for idx, item in enumerate(samples, start=1):
                src = str(item.get("source", "")).replace("|", "\\|")
                tgt = str(item.get("target", "")).replace("|", "\\|")
                pred = str(item.get("prediction", "")).replace("|", "\\|")
                beam = item.get("beam_size", 1)
                md_lines.append(f"| {idx} | {src} | {tgt} | {pred} | {beam} |")

            md_content = "\n".join(md_lines)

            with tempfile.NamedTemporaryFile("w", suffix=f"_epoch_{epoch}.md", delete=False, encoding="utf-8") as tmp:
                tmp.write(md_content)
                tmp_name = tmp.name

            try:
                mlflow.log_artifact(tmp_name, artifact_path=artifact_path)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)

        except Exception as e:
            logger.warning(f"Failed to log translation samples at epoch {epoch}: {e}")
