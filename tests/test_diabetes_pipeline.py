"""Tests for diabetes feature engineering and preprocessing flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.paths import get_task_manifest
from src.features.diabetes_features import add_diabetes_engineered_features, replace_impossible_zeros
from src.preprocessing.diabetes_preprocess import prepare_diabetes_dataframe, transform_diabetes_for_inference


def test_diabetes_feature_engineering_creates_expected_columns() -> None:
    task = get_task_manifest("diabetes")
    payload = {
        "Pregnancies": 2.0,
        "Glucose": 150.0,
        "BloodPressure": 88.0,
        "SkinThickness": 25.0,
        "Insulin": 130.0,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.65,
        "Age": 50.0,
    }
    raw_df = prepare_diabetes_dataframe(payload, task["raw_features"])
    transformed = replace_impossible_zeros(raw_df)
    featured = add_diabetes_engineered_features(transformed)

    for column in task["inference_columns"]:
        assert column in featured.columns

    assert featured.loc[0, "Age_Glucose"] == 7500.0
    assert featured.loc[0, "High_Glucose"] == 1


def test_diabetes_preprocessing_applies_imputation_and_scaling() -> None:
    task = get_task_manifest("diabetes")
    payload = {
        "Pregnancies": 1.0,
        "Glucose": 0.0,
        "BloodPressure": 80.0,
        "SkinThickness": 0.0,
        "Insulin": 0.0,
        "BMI": 0.0,
        "DiabetesPedigreeFunction": 0.3,
        "Age": 35.0,
    }
    raw_df = prepare_diabetes_dataframe(payload, task["raw_features"])

    inference_columns = task["inference_columns"]
    fit_df = pd.DataFrame(
        [
            np.linspace(1, 19, 19),
            np.linspace(2, 20, 19),
            np.linspace(3, 21, 19),
        ],
        columns=inference_columns,
    )
    imputer = SimpleImputer(strategy="mean").fit(fit_df)
    scaler = StandardScaler().fit(fit_df)

    imputed_df, scaled_df = transform_diabetes_for_inference(
        raw_dataframe=raw_df,
        imputer=imputer,
        scaler=scaler,
        inference_columns=inference_columns,
    )

    assert imputed_df.shape == (1, len(inference_columns))
    assert scaled_df.shape == (1, len(inference_columns))
    assert not np.isnan(imputed_df.values).any()
