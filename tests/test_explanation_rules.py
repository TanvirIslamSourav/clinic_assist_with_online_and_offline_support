"""Tests for deterministic explanation rules."""

from __future__ import annotations

from src.explainability.explanation_rules import (
    generate_image_explanation,
    generate_tabular_explanation,
    get_confidence_band,
)


def test_confidence_band_mapping() -> None:
    assert get_confidence_band(0.9) == "high confidence"
    assert get_confidence_band(0.7) == "moderate confidence"
    assert get_confidence_band(0.5) == "lower confidence"


def test_tabular_explanation_contains_required_safety_language() -> None:
    text = generate_tabular_explanation(
        task_name="Diabetes Risk Prediction",
        predicted_label="Elevated Diabetes Risk",
        predicted_probability=0.82,
        confidence_band="moderate confidence",
        positive_contributors=["BMI_Glucose", "Glucose", "Age_Glucose"],
        negative_contributors=["Insulin_Glucose"],
    )
    lowered = text.lower()
    assert "diabetes risk prediction" in lowered
    assert "model behavior" in lowered
    assert "not a final diagnosis" in lowered


def test_image_explanation_mentions_attention_only() -> None:
    text = generate_image_explanation(
        predicted_label="PNEUMONIA",
        predicted_probability=0.91,
        confidence_band="high confidence",
    )
    lowered = text.lower()
    assert "grad-cam" in lowered
    assert "model attention only" in lowered
