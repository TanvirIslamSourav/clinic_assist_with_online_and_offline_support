"""Tests for safe SHAP runtime behavior."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import src.explainability.shap_runtime as shap_runtime


class DummyModel:
    def predict_proba(self, dataframe: pd.DataFrame):
        return [[0.40, 0.60] for _ in range(len(dataframe))]


def test_generate_local_shap_handles_missing_dependency(monkeypatch) -> None:
    def _raise_import_error() -> None:
        raise ImportError("shap missing")

    monkeypatch.setattr(shap_runtime, "_import_shap_module",
                        _raise_import_error)

    result = shap_runtime.generate_local_shap_figure(
        task_key="diabetes",
        model=DummyModel(),
        model_input=pd.DataFrame(
            [{
                "Pregnancies": 2.0,
                "Glucose": 150.0,
                "BloodPressure": 88.0,
                "SkinThickness": 25.0,
                "Insulin": 120.0,
                "BMI": 31.0,
                "DiabetesPedigreeFunction": 0.7,
                "Age": 50.0,
                "Age_Glucose": 7500.0,
                "BMI_Insulin": 3720.0,
                "Glucose_Preg": 300.0,
                "BMI_Glucose": 4650.0,
                "Glucose_sq": 22500.0,
                "BMI_sq": 961.0,
                "Insulin_Glucose": 18000.0,
                "Age_group": 1.0,
                "High_Glucose": 1.0,
                "Obese": 1.0,
                "High_BP": 1.0,
            }]
        ),
    )

    assert result.success is True
    assert result.figure is not None
    assert "SHAP dependency is not available" in (result.error or "")
    plt.close(result.figure)
