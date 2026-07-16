"""Tests for explainability fallback image selection."""

from __future__ import annotations

from pathlib import Path

from src.explainability.fallback_loader import (
    first_existing_path,
    get_tabular_fallback_candidates,
)


def test_first_existing_path_returns_first_valid_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"
    fallback_path = tmp_path / "fallback.png"
    fallback_path.write_bytes(b"fake-image")

    selected = first_existing_path([missing_path, fallback_path])

    assert selected == fallback_path


def test_diabetes_fallback_candidates_include_expected_shap_assets() -> None:
    candidates = get_tabular_fallback_candidates("diabetes")
    candidate_names = [path.name for path in candidates]

    assert "shap_waterfall_Diabetes_s1.png" in candidate_names
    assert "shap_bar_Diabetes.png" in candidate_names
