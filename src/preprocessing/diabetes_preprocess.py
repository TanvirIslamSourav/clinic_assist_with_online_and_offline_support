"""Diabetes preprocessing service preserving notebook inference path."""

from __future__ import annotations

import pandas as pd

from src.features.diabetes_features import add_diabetes_engineered_features, replace_impossible_zeros


def prepare_diabetes_dataframe(raw_payload: dict[str, float], expected_raw_features: list[str]) -> pd.DataFrame:
    """Create one-row dataframe from form payload in expected raw feature order."""
    row = {feature: raw_payload[feature] for feature in expected_raw_features}
    return pd.DataFrame([row])


def transform_diabetes_for_inference(
    raw_dataframe: pd.DataFrame,
    imputer,
    scaler,
    inference_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply zero replacement, feature engineering, imputation, scaling, and column ordering."""
    transformed = replace_impossible_zeros(raw_dataframe)
    transformed = add_diabetes_engineered_features(transformed)

    transformed = transformed.reindex(columns=inference_columns)
    imputed_array = imputer.transform(transformed)
    imputed_df = pd.DataFrame(
        imputed_array, columns=inference_columns, index=transformed.index)

    scaled_array = scaler.transform(imputed_df)
    scaled_df = pd.DataFrame(
        scaled_array, columns=inference_columns, index=imputed_df.index)
    return imputed_df, scaled_df
