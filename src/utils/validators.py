"""Validation helpers for tabular and image inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def coerce_numeric_payload(payload: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Convert payload values to floats while collecting conversion errors."""
    parsed: dict[str, float] = {}
    errors: list[str] = []
    for key, value in payload.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid numeric value for '{key}': {value}")
    return parsed, errors


def validate_expected_columns(dataframe: pd.DataFrame, expected_columns: list[str]) -> tuple[bool, str]:
    """Ensure required columns are present in a dataframe."""
    missing = [
        column for column in expected_columns if column not in dataframe.columns]
    if missing:
        return False, f"Missing expected columns: {', '.join(missing)}"
    return True, ""


def validate_uploaded_filename(filename: str | None) -> tuple[bool, str]:
    """Validate uploaded image filename extension."""
    if not filename:
        return False, "No file uploaded."
    lower_name = filename.lower()
    allowed = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not lower_name.endswith(allowed):
        return False, "Unsupported image format. Please upload PNG/JPG/BMP/TIFF."
    return True, ""
