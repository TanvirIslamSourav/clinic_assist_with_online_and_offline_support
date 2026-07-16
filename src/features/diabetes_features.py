"""Diabetes feature engineering aligned with notebook inference logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZERO_TO_NAN_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def replace_impossible_zeros(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zero values with NaN for clinical variables."""
    transformed = dataframe.copy()
    transformed[ZERO_TO_NAN_COLUMNS] = transformed[ZERO_TO_NAN_COLUMNS].replace(
        0, np.nan)
    return transformed


def add_diabetes_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features exactly as specified by notebook logic."""
    engineered = dataframe.copy()

    engineered["Age_Glucose"] = engineered["Age"] * engineered["Glucose"]
    engineered["BMI_Insulin"] = engineered["BMI"] * engineered["Insulin"]
    engineered["Glucose_Preg"] = engineered["Glucose"] * \
        engineered["Pregnancies"]
    engineered["BMI_Glucose"] = engineered["BMI"] * engineered["Glucose"]
    engineered["Glucose_sq"] = engineered["Glucose"] ** 2
    engineered["BMI_sq"] = engineered["BMI"] ** 2
    engineered["Insulin_Glucose"] = engineered["Insulin"] / \
        (engineered["Glucose"] + 1e-9)

    engineered["Age_group"] = pd.cut(
        engineered["Age"],
        bins=[20, 30, 40, 50, 60, 90],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )
    engineered["Age_group"] = pd.to_numeric(
        engineered["Age_group"], errors="coerce")

    engineered["High_Glucose"] = (engineered["Glucose"] > 140).astype(int)
    engineered["Obese"] = (engineered["BMI"] > 30).astype(int)
    engineered["High_BP"] = (engineered["BloodPressure"] > 90).astype(int)

    return engineered
