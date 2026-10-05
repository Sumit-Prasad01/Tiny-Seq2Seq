"""
Unit tests for Phase 1: MLflow Experiment Tracking Engine.
"""

import os
import shutil
import tempfile
import mlflow
import pytest

from src.seq2seq.training.tracker import MLflowTracker, _flatten_dict


@pytest.fixture
def temp_mlflow_env():
    """Provides an isolated temporary tracking directory for MLflow tests."""
    temp_dir = tempfile.mkdtemp()
    sqlite_db = os.path.join(temp_dir, "test_mlruns.db").replace("\\", "/")
    tracking_uri = f"sqlite:///{sqlite_db}"
    yield tracking_uri, temp_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_flatten_dict():
    nested = {
        "model": {"d_model": 512, "n_layers": 3},
        "training": {"lr": 1e-3, "optimizer": "Adam"},
        "tags": ["seq2seq", "wmt14"],
    }
    flat = _flatten_dict(nested)
    assert flat["model.d_model"] == 512
    assert flat["model.n_layers"] == 3
    assert flat["training.lr"] == 1e-3
    assert flat["tags"] == "['seq2seq', 'wmt14']"


def test_tracker_lifecycle_and_metrics(temp_mlflow_env):
    tracking_uri, _ = temp_mlflow_env
    tracker = MLflowTracker(
        experiment_name="Test-Experiment-Lifecycle",
        tracking_uri=tracking_uri,
    )

    params = {
        "d_model": 512,
        "n_layers": 3,
        "dropout": 0.2,
        "optimizer": {"name": "Adam", "lr": 0.001},
    }

    with tracker.start_run(run_name="test_run_1", tags={"env": "test"}) as run:
        run_id = run.info.run_id
        tracker.log_params(params)

        # Log step metrics
        for step in range(5):
            tracker.log_step_metrics(
                step=step,
                metrics={
                    "step/train_loss": 5.0 - step * 0.5,
                    "step/lr": 0.001,
                    "step/vram_allocated_mb": 1200.5,
                },
            )

        # Log epoch metrics
        tracker.log_epoch_metrics(
            epoch=1,
            metrics={"epoch/train_loss": 3.0, "epoch/dev_loss": 3.5, "epoch/dev_bleu": 12.4},
        )

        # Log translation samples table
        samples = [
            {"source": "Hello world", "target": "Bonjour le monde", "prediction": "Bonjour monde", "beam_size": 5},
            {"source": "Machine translation", "target": "Traduction automatique", "prediction": "Traduction automatique", "beam_size": 5},
        ]
        tracker.log_translation_samples(samples=samples, epoch=1)

    # Query back using MLflow Client to verify
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(run_id)

    assert run_data.data.params["d_model"] == "512"
    assert run_data.data.params["optimizer.name"] == "Adam"
    assert "epoch/dev_bleu" in run_data.data.metrics
    assert run_data.data.metrics["epoch/dev_bleu"] == 12.4

    # Verify artifact was logged
    artifacts = client.list_artifacts(run_id, path="translations")
    assert len(artifacts) >= 1
    assert artifacts[0].path.startswith("translations")


def test_nested_runs_for_ablations(temp_mlflow_env):
    tracking_uri, _ = temp_mlflow_env
    tracker = MLflowTracker(
        experiment_name="Test-Ablations",
        tracking_uri=tracking_uri,
    )

    with tracker.start_run(run_name="parent_ablation_study") as parent_run:
        parent_id = parent_run.info.run_id
        tracker.log_params({"study": "depth_ablation"})

        # Child run 1
        with tracker.start_run(run_name="depth_2_layers", nested=True) as child1:
            tracker.log_params({"n_layers": 2})
            tracker.log_epoch_metrics(epoch=1, metrics={"bleu": 10.5})

        # Child run 2
        with tracker.start_run(run_name="depth_3_layers", nested=True) as child2:
            tracker.log_params({"n_layers": 3})
            tracker.log_epoch_metrics(epoch=1, metrics={"bleu": 13.2})

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    child1_data = client.get_run(child1.info.run_id)
    assert child1_data.data.tags.get("mlflow.parentRunId") == parent_id
