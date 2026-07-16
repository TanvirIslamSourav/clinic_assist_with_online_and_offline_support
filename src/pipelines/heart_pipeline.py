"""Heart disease inference pipeline with pre-pipeline feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.explainability.explanation_rules import get_confidence_band
from src.models.heart_model_loader import load_heart_artifacts
from src.preprocessing.heart_preprocess import prepare_heart_dataframe
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class HeartPredictionResult:
    label: str
    probability: float
    confidence_band: str
    positive_contributors: list[str]
    negative_contributors: list[str]
    model_input: pd.DataFrame


def _align_with_pipeline_expected_columns(dataframe: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Align input dataframe with pipeline expected feature names when available."""
    expected_columns = getattr(pipeline, "feature_names_in_", None)
    if expected_columns is None:
        return dataframe

    expected_list = list(expected_columns)
    adjusted_df = dataframe.copy()
    alias_pairs = {
        "thalch": "thalach",
        "thalach": "thalch",
    }
    for expected_col, alias_col in alias_pairs.items():
        if expected_col in expected_list and expected_col not in adjusted_df.columns and alias_col in adjusted_df.columns:
            adjusted_df[expected_col] = adjusted_df[alias_col]

    missing = [col for col in expected_list if col not in adjusted_df.columns]
    if missing:
        raise ValueError(
            f"Heart inference failed: missing engineered columns expected by pipeline: {missing}")
    return adjusted_df.reindex(columns=expected_list)


def _extract_heart_contributors(row: pd.Series) -> tuple[list[str], list[str]]:
    """Derive rule-based contributor ranking from clinical feature patterns."""
    scores: dict[str, float] = {
        "age": (row["age"] - 50.0) / 15.0,
        "trestbps": (row["trestbps"] - 130.0) / 20.0,
        "chol": (row["chol"] - 210.0) / 35.0,
        "oldpeak": (row["oldpeak"] - 1.0) / 1.0,
        "exang": float(row["exang"]),
        "ca": float(row["ca"]) / 2.0,
        "hr_reserve": -(row["hr_reserve"]) / 40.0,
        "thalach": -(row["thalach"] - 140.0) / 35.0,
        "chol_high": float(row["chol_high"]),
        "bp_high": float(row["bp_high"]),
        "senior": float(row["senior"]),
    }

    positive = sorted([item for item in scores.items()
                      if item[1] > 0], key=lambda x: x[1], reverse=True)
    negative = sorted([item for item in scores.items()
                      if item[1] < 0], key=lambda x: x[1])

    positive_features = [name for name, _ in positive[:4]]
    negative_features = [name for name, _ in negative[:3]]
    return positive_features, negative_features


def predict_heart(raw_payload: dict[str, float]) -> HeartPredictionResult:
    """Run heart disease inference from raw form payload and engineered features."""
    artifacts = load_heart_artifacts()
    task = artifacts["task_manifest"]
    pipeline = artifacts["pipeline"]

    if pipeline is None:
        raw_df = prepare_heart_dataframe(raw_payload, task["raw_features"])
        LOGGER.warning("Heart pipeline artifact missing; returning degraded result.")
        return HeartPredictionResult(
            label="Unavailable",
            probability=0.0,
            confidence_band="unknown",
            positive_contributors=[],
            negative_contributors=[],
            model_input=raw_df,
        )

    engineered_df = prepare_heart_dataframe(raw_payload, task["raw_features"])
    model_input = _align_with_pipeline_expected_columns(
        engineered_df, pipeline)

    probabilities = pipeline.predict_proba(model_input)
    positive_prob = float(
        probabilities[0, 1]) if probabilities.shape[1] > 1 else float(probabilities[0, 0])
    predicted_index = int(positive_prob >= 0.5)
    confidence = get_confidence_band(positive_prob)

    positive_contributors, negative_contributors = _extract_heart_contributors(
        engineered_df.iloc[0])

    LOGGER.info("Heart prediction complete with probability %.4f", positive_prob)
    return HeartPredictionResult(
        label=task["labels"][predicted_index],
        probability=float(np.clip(positive_prob, 0.0, 1.0)),
        confidence_band=confidence,
        positive_contributors=positive_contributors,
        negative_contributors=negative_contributors,
        model_input=model_input,
    )
