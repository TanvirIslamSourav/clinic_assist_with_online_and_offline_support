"""Loader for heart disease pipeline artifact."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from config.paths import get_task_manifest, resolve_path
from src.utils.file_utils import ensure_file_exists
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_heart_artifacts() -> dict[str, Any]:
    """Load heart sklearn pipeline using manifest-defined path."""
    task = get_task_manifest("heart")
    model_path = resolve_path(task["model_path"])

    if not model_path.exists():
        LOGGER.warning(
            "Heart pipeline artifact not found at %s; running in degraded mode.",
            model_path,
        )
        return {
            "pipeline": None,
            "task_manifest": task,
            "model_path": model_path,
            "degraded_reason": "model_file_missing",
        }

    try:
        pipeline = joblib.load(model_path)
    except AttributeError as exc:
        message = str(exc)
        if "_RemainderColsList" in message:
            LOGGER.exception(
                "Failed loading heart pipeline due to scikit-learn version mismatch.")
            raise RuntimeError(
                "Failed to load heart model artifact. The saved pipeline was built with a different "
                "scikit-learn version. Install scikit-learn==1.6.1 and retry."
            ) from exc
        LOGGER.exception("Failed loading heart pipeline artifact.")
        raise RuntimeError("Failed to load heart model artifact.") from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading heart pipeline artifact.")
        raise RuntimeError("Failed to load heart model artifact.") from exc

    LOGGER.info("Loaded heart pipeline successfully.")
    return {
        "pipeline": pipeline,
        "task_manifest": task,
    }
