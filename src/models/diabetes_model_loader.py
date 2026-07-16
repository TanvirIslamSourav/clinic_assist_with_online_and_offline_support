"""Loader for diabetes model and preprocessing artifacts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from config.paths import get_task_manifest, resolve_path
from src.utils.file_utils import ensure_file_exists
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_diabetes_artifacts() -> dict[str, Any]:
    """Load diabetes model, imputer, and scaler artifacts from manifest paths."""
    task = get_task_manifest("diabetes")

    model_path = resolve_path(task["model_path"])
    imputer_path = resolve_path(task["preprocessors"]["imputer_path"])
    scaler_path = resolve_path(task["preprocessors"]["scaler_path"])

    if not (model_path.exists() and imputer_path.exists() and scaler_path.exists()):
        LOGGER.warning(
            "Diabetes model artifacts not found; running in degraded mode. "
            "Paths checked: %s, %s, %s",
            model_path, imputer_path, scaler_path,
        )
        return {
            "model": None,
            "imputer": None,
            "scaler": None,
            "task_manifest": task,
            "model_path": model_path,
            "degraded_reason": "model_files_missing",
        }

    try:
        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or str(exc)
        LOGGER.exception(
            "Failed loading diabetes artifacts due to missing dependency module: %s",
            missing_module,
        )
        raise RuntimeError(
            "Failed to load diabetes model artifacts. "
            f"Missing dependency: {missing_module}. "
            "Install it in your active environment and retry."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading diabetes artifacts.")
        raise RuntimeError("Failed to load diabetes model artifacts.") from exc

    LOGGER.info("Loaded diabetes artifacts successfully.")
    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "task_manifest": task,
    }
