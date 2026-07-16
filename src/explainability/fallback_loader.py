"""Fallback image selection helpers for explainability figures."""

from __future__ import annotations

from pathlib import Path

from config.paths import get_task_manifest, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_TABULAR_PRIORITY_OVERRIDES: dict[str, list[str]] = {
    "diabetes": [
        "static_outputs/diabetes/shap_patient_contributions_Diabetes.png",
        "static_outputs/diabetes/shap_waterfall_Diabetes_s1.png",
        "static_outputs/diabetes/shap_bar_Diabetes.png",
        "static_outputs/diabetes/lime_Diabetes_HighRisk_Positive.png",
    ],
    "heart": [
        "static_outputs/heart/shap_patient_contributions_Heart.png",
        "static_outputs/heart/shap_waterfall_Heart_s1.png",
        "static_outputs/heart/shap_bar_Heart.png",
        "static_outputs/heart/lime_Heart_HighRisk_Positive.png",
    ],
}


def first_existing_path(candidates: list[Path]) -> Path | None:
    """Return first existing image path from candidates."""
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def get_tabular_fallback_candidates(task_key: str) -> list[Path]:
    """Get ordered static fallback candidates for a tabular task."""
    task_manifest = get_task_manifest(task_key)
    configured_xai_paths = task_manifest.get(
        "static_figures", {}).get("xai", [])

    priority_paths = _TABULAR_PRIORITY_OVERRIDES.get(task_key, [])
    merged_paths = priority_paths + [
        path
        for path in configured_xai_paths
        if path not in priority_paths
    ]
    return [resolve_path(path) for path in merged_paths]


def get_tabular_fallback_image(task_key: str) -> Path | None:
    """Resolve best available static fallback explanation image for a tabular task."""
    candidates = get_tabular_fallback_candidates(task_key)
    selected = first_existing_path(candidates)

    if selected is not None:
        LOGGER.info(
            "Using static explanation fallback for task=%s: %s",
            task_key,
            selected.name,
        )
    else:
        LOGGER.warning(
            "No static explanation fallback images found for task=%s",
            task_key,
        )

    return selected
