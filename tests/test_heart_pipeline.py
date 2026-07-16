"""Tests for heart feature engineering logic."""

from __future__ import annotations

import pandas as pd

from config.paths import get_task_manifest
from src.features.heart_features import add_heart_engineered_features
from src.pipelines.heart_pipeline import _align_with_pipeline_expected_columns
from src.preprocessing.heart_preprocess import prepare_heart_dataframe


def test_heart_feature_engineering_values() -> None:
    base = {
        "age": 60.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 145.0,
        "chol": 260.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 130.0,
        "exang": 1.0,
        "oldpeak": 2.0,
        "slope": 1.0,
        "ca": 1.0,
        "thal": 2.0,
    }
    dataframe = prepare_heart_dataframe(
        base, get_task_manifest("heart")["raw_features"])
    engineered = add_heart_engineered_features(dataframe)

    assert engineered.loc[0, "age_maxhr"] == 7800.0
    assert engineered.loc[0, "bp_high"] == 1
    assert engineered.loc[0, "chol_high"] == 1
    assert engineered.loc[0, "oldpeak_hr"] == 260.0


def test_prepare_heart_dataframe_contains_engineered_columns() -> None:
    task = get_task_manifest("heart")
    payload = {
        "age": 45.0,
        "sex": 0.0,
        "cp": 1.0,
        "trestbps": 120.0,
        "chol": 190.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 165.0,
        "exang": 0.0,
        "oldpeak": 0.6,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 2.0,
    }
    dataframe = prepare_heart_dataframe(payload, task["raw_features"])
    for feature in task["engineered_features"]:
        assert feature in dataframe.columns


def test_align_expected_columns_handles_thalch_alias() -> None:
    class DummyPipeline:
        feature_names_in_ = ["age", "thalch", "oldpeak_hr"]

    source = pd.DataFrame(
        [{"age": 55.0, "thalach": 145.0, "oldpeak_hr": 116.0}])

    aligned = _align_with_pipeline_expected_columns(source, DummyPipeline())

    assert list(aligned.columns) == ["age", "thalch", "oldpeak_hr"]
    assert aligned.loc[0, "thalch"] == 145.0
