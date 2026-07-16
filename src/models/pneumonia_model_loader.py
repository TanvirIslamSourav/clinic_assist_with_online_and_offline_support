"""Loader for pneumonia DenseNet Keras model."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    import tensorflow as tf  # noqa: E402
except ImportError:  # Tensorflow is optional; pneumonia module runs degraded.
    tf = None  # type: ignore[assignment]

from config.paths import get_task_manifest, resolve_path
from src.utils.file_utils import ensure_file_exists
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_pneumonia_artifacts() -> dict[str, Any]:
    """Load Keras model for pneumonia detection from configured path."""
    task = get_task_manifest("pneumonia")
    model_path = resolve_path(task["model_path"])

    if tf is None:
        LOGGER.warning(
            "TensorFlow is not installed; pneumonia module will run in degraded mode."
        )
        return {
            "model": None,
            "task_manifest": task,
            "model_path": model_path,
            "degraded_reason": "tensorflow_not_installed",
        }

    if not model_path.exists():
        return {
            "model": None,
            "task_manifest": task,
            "model_path": model_path,
            "degraded_reason": "model_file_missing",
        }

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading pneumonia model artifact.")
        raise RuntimeError("Failed to load pneumonia model artifact.") from exc

    LOGGER.info("Loaded pneumonia model successfully.")
    return {
        "model": model,
        "task_manifest": task,
    }
