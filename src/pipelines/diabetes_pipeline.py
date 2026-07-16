"""Diabetes inference pipeline preserving notebook-time preprocessing flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.explainability.explanation_rules import get_confidence_band
from src.models.diabetes_model_loader import load_diabetes_artifacts
from src.preprocessing.diabetes_preprocess import prepare_diabetes_dataframe, transform_diabetes_for_inference
from src.utils.logger import get_logger
from src.utils.validators import validate_expected_columns

LOGGER = get_logger(__name__)


@dataclass
class DiabetesPredictionResult:
    label: str
    probability: float
    confidence_band: str
    positive_contributors: list[str]
    negative_contributors: list[str]
    imputed_features: pd.DataFrame
    scaled_features: pd.DataFrame


def _extract_diabetes_contributors(row: pd.Series) -> tuple[list[str], list[str]]:
    """Derive simple rule-based contributor ranking from imputed feature values."""
    scores: dict[str, float] = {
        "Glucose": (row["Glucose"] - 110.0) / 25.0,
        "BMI": (row["BMI"] - 27.0) / 8.0,
        "Age": (row["Age"] - 45.0) / 15.0,
        "BloodPressure": (row["BloodPressure"] - 85.0) / 15.0,
        "Age_Glucose": (row["Age_Glucose"] - 5000.0) / 2500.0,
        "BMI_Glucose": (row["BMI_Glucose"] - 3000.0) / 1000.0,
        "High_Glucose": float(row["High_Glucose"]),
        "Obese": float(row["Obese"]),
        "High_BP": float(row["High_BP"]),
    }

    positive = sorted([item for item in scores.items()
                      if item[1] > 0], key=lambda x: x[1], reverse=True)
    negative = sorted([item for item in scores.items()
                      if item[1] < 0], key=lambda x: x[1])

    positive_features = [name for name, _ in positive[:4]]
    negative_features = [name for name, _ in negative[:3]]
    return positive_features, negative_features


def predict_diabetes(raw_payload: dict[str, float]) -> DiabetesPredictionResult:
    """Run diabetes inference from raw form payload using saved notebook artifacts."""
    artifacts = load_diabetes_artifacts()
    task = artifacts["task_manifest"]
    model = artifacts["model"]

    if model is None:
        # Degraded mode — produce an empty result so the UI can show an explanation.
        raw_df = prepare_diabetes_dataframe(raw_payload, task["raw_features"])
        empty_df = raw_df.copy()
        LOGGER.warning("Diabetes model artifact missing; returning degraded result.")
        return DiabetesPredictionResult(
            label="Unavailable",
            probability=0.0,
            confidence_band="unknown",
            positive_contributors=[],
            negative_contributors=[],
            imputed_features=empty_df,
            scaled_features=empty_df,
        )

    raw_df = prepare_diabetes_dataframe(raw_payload, task["raw_features"])
    imputed_df, scaled_df = transform_diabetes_for_inference(
        raw_dataframe=raw_df,
        imputer=artifacts["imputer"],
        scaler=artifacts["scaler"],
        inference_columns=task["inference_columns"],
    )

    valid, message = validate_expected_columns(
        scaled_df, task["inference_columns"])
    if not valid:
        raise ValueError(message)

    probabilities = model.predict_proba(scaled_df)
    positive_prob = float(
        probabilities[0, 1]) if probabilities.shape[1] > 1 else float(probabilities[0, 0])
    predicted_index = int(positive_prob >= 0.5)

    confidence = get_confidence_band(positive_prob)
    positive_contributors, negative_contributors = _extract_diabetes_contributors(
        imputed_df.iloc[0])

    LOGGER.info(
        "Diabetes prediction complete with probability %.4f", positive_prob)
    return DiabetesPredictionResult(
        label=task["labels"][predicted_index],
        probability=float(np.clip(positive_prob, 0.0, 1.0)),
        confidence_band=confidence,
        positive_contributors=positive_contributors,
        negative_contributors=negative_contributors,
        imputed_features=imputed_df,
        scaled_features=scaled_df,
    )
