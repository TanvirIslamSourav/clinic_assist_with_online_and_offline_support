"""Runtime SHAP graph generation for tabular tasks with safe fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explainability.feature_name_mapping import humanize_feature
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Reference rows are only used as SHAP background maskers; prediction still uses
# the exact transformed input produced by existing inference pipelines.
_BASELINE_ROWS: dict[str, dict[str, float]] = {
    "diabetes": {
        "Pregnancies": 1.0,
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "SkinThickness": 20.0,
        "Insulin": 80.0,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45.0,
        "Age_Glucose": 5400.0,
        "BMI_Insulin": 2240.0,
        "Glucose_Preg": 120.0,
        "BMI_Glucose": 3360.0,
        "Glucose_sq": 14400.0,
        "BMI_sq": 784.0,
        "Insulin_Glucose": 9600.0,
        "Age_group": 1.0,
        "High_Glucose": 1.0,
        "Obese": 1.0,
        "High_BP": 0.0,
    },
    "heart": {
        "age": 55.0,
        "sex": 1.0,
        "cp": 0.0,
        "trestbps": 130.0,
        "chol": 240.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 150.0,
        "thalch": 150.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 2.0,
        "age_maxhr": 8250.0,
        "hr_reserve": 15.0,
        "bp_high": 0.0,
        "bp_low": 0.0,
        "chol_high": 1.0,
        "age_sq": 3025.0,
        "senior": 0.0,
        "oldpeak_hr": 150.0,
    },
}


@dataclass
class ShapRuntimeResult:
    """Result object for optional SHAP graph generation."""

    success: bool
    figure: Any | None = None
    error: str | None = None


def _import_shap_module() -> Any:
    """Import SHAP lazily to avoid hard runtime dependency when unavailable."""
    import shap

    return shap


def _coerce_numeric_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    coerced = dataframe.copy()
    for column in coerced.columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced.fillna(0.0)


def _looks_like_standardized_features(model_input: pd.DataFrame) -> bool:
    values = np.asarray(model_input, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    within_standard_range = np.mean(np.abs(finite_values) <= 6.0)
    return bool(within_standard_range >= 0.85)


def _build_background_frame(task_key: str, model_input: pd.DataFrame) -> pd.DataFrame:
    baseline_values = _BASELINE_ROWS.get(task_key, {})
    if _looks_like_standardized_features(model_input):
        baseline_row = pd.Series(0.0, index=model_input.columns, dtype=float)
    else:
        baseline_row = model_input.iloc[0].copy()

        for column in baseline_row.index:
            if column in baseline_values:
                baseline_row[column] = float(baseline_values[column])

    background = pd.concat(
        [baseline_row.to_frame().T, model_input.copy()],
        axis=0,
        ignore_index=True,
    )
    return _coerce_numeric_frame(background)


def _build_predict_proba_fn(model: Any, feature_names: list[str]) -> Callable[[Any], np.ndarray]:
    def _predict_positive_probability(data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            frame = data.reindex(columns=feature_names, fill_value=0.0)
        else:
            array = np.asarray(data, dtype=float)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            frame = pd.DataFrame(array, columns=feature_names)

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            if probabilities.ndim == 2:
                if probabilities.shape[1] > 1:
                    return probabilities[:, 1]
                return probabilities[:, 0]
            return probabilities.reshape(-1)

        predictions = np.asarray(model.predict(frame), dtype=float)
        return predictions.reshape(-1)

    return _predict_positive_probability


def _extract_local_values(shap_values: Any) -> np.ndarray:
    values = np.asarray(getattr(shap_values, "values"))

    if values.ndim == 1:
        return values.astype(float)
    if values.ndim == 2:
        return values[0].astype(float)
    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        return values[0, :, class_index].astype(float)

    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def _extract_kernel_values(kernel_values: Any) -> np.ndarray:
    if isinstance(kernel_values, list):
        if not kernel_values:
            raise ValueError("KernelExplainer returned empty SHAP list.")
        class_index = 1 if len(kernel_values) > 1 else 0
        values = np.asarray(kernel_values[class_index], dtype=float)
    else:
        values = np.asarray(kernel_values, dtype=float)

    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        return values[0, :, class_index]

    raise ValueError(f"Unsupported KernelExplainer SHAP shape: {values.shape}")


def _build_contribution_figure(
    feature_names: list[str],
    shap_vector: np.ndarray,
    max_features: int,
    title: str,
) -> Any:
    if shap_vector.size != len(feature_names):
        raise ValueError(
            "SHAP feature vector length does not match model input columns: "
            f"{shap_vector.size} vs {len(feature_names)}"
        )

    ordered_indices = np.argsort(np.abs(shap_vector))[
        ::-1][: max(1, max_features)]
    selected_values = shap_vector[ordered_indices]
    selected_names = [humanize_feature(
        feature_names[index]) for index in ordered_indices]

    colors = ["#b91c1c" if value >=
              0 else "#047857" for value in selected_values]
    figure_height = max(3.2, 1.8 + 0.45 * len(selected_values))
    fig, ax = plt.subplots(figsize=(8.0, figure_height))

    y_positions = np.arange(len(selected_values))
    ax.barh(y_positions, selected_values, color=colors, alpha=0.9)
    ax.set_yticks(y_positions, labels=selected_names)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_xlabel("SHAP contribution to positive-class risk")
    ax.set_title(title)

    for y_pos, value in zip(y_positions, selected_values):
        alignment = "left" if value >= 0 else "right"
        x_offset = 0.01 if value >= 0 else -0.01
        ax.text(value + x_offset, y_pos,
                f"{value:+.3f}", va="center", ha=alignment, fontsize=9)

    fig.tight_layout()
    return fig


def _estimate_local_contributions_by_ablation(
    predict_fn: Callable[[Any], np.ndarray],
    model_input: pd.DataFrame,
    reference_row: pd.Series,
    feature_names: list[str],
) -> np.ndarray:
    """Approximate local feature impact by replacing one feature at a time."""
    base_prob = float(np.asarray(predict_fn(
        model_input), dtype=float).reshape(-1)[0])
    contributions: list[float] = []

    for feature in feature_names:
        ablated = model_input.copy()
        ablated.loc[ablated.index[0], feature] = float(reference_row[feature])
        ablated_prob = float(np.asarray(
            predict_fn(ablated), dtype=float).reshape(-1)[0])
        contributions.append(base_prob - ablated_prob)

    return np.asarray(contributions, dtype=float)


def generate_local_shap_figure(
    task_key: str,
    model: Any,
    model_input: pd.DataFrame,
    max_features: int = 10,
) -> ShapRuntimeResult:
    """Try generating a local SHAP contribution chart for one prediction input."""
    try:
        if model_input is None or model_input.empty:
            raise ValueError("Model input for SHAP explanation is empty.")

        model_input_df = _coerce_numeric_frame(model_input.copy())
        feature_names = list(model_input_df.columns)
        LOGGER.info(
            "Starting SHAP explanation generation for task=%s", task_key)

        background = _build_background_frame(task_key, model_input_df)
        predict_fn = _build_predict_proba_fn(model, feature_names)

        local_values: np.ndarray | None = None
        figure_title = "Local SHAP Feature Contributions"
        shap: Any | None = None
        shap_import_error: Exception | None = None

        try:
            shap = _import_shap_module()
        except Exception as exc:  # noqa: BLE001
            shap_import_error = exc
            LOGGER.warning(
                "SHAP dependency is unavailable for task=%s; using local ablation approximation.",
                task_key,
            )

        if shap is None:
            local_values = _estimate_local_contributions_by_ablation(
                predict_fn=predict_fn,
                model_input=model_input_df,
                reference_row=background.iloc[0],
                feature_names=feature_names,
            )
            figure_title = "Local Feature Contribution Approximation"
        else:
            try:
                explainer = shap.Explainer(
                    predict_fn,
                    background,
                    feature_names=feature_names,
                )
                call_kwargs = {"max_evals": max(
                    33, 2 * len(feature_names) + 1)}
                try:
                    shap_values = explainer(model_input_df, **call_kwargs)
                except TypeError:
                    shap_values = explainer(model_input_df)
                local_values = _extract_local_values(shap_values)
            except Exception as first_exc:  # noqa: BLE001
                LOGGER.warning(
                    "Primary SHAP explainer failed for task=%s; trying KernelExplainer fallback. Error=%s",
                    task_key,
                    first_exc,
                )
                try:
                    kernel_explainer = shap.KernelExplainer(
                        predict_fn,
                        background.to_numpy(dtype=float),
                    )
                    kernel_values = kernel_explainer.shap_values(
                        model_input_df.to_numpy(dtype=float),
                        nsamples=max(33, 2 * len(feature_names) + 1),
                    )
                    local_values = _extract_kernel_values(kernel_values)
                except Exception as kernel_exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Kernel SHAP failed for task=%s; using local ablation approximation. Error=%s",
                        task_key,
                        kernel_exc,
                    )
                    local_values = _estimate_local_contributions_by_ablation(
                        predict_fn=predict_fn,
                        model_input=model_input_df,
                        reference_row=background.iloc[0],
                        feature_names=feature_names,
                    )
                    figure_title = "Local Feature Contribution Approximation"

        if local_values is None:
            raise RuntimeError("SHAP produced no local contribution values.")

        figure = _build_contribution_figure(
            feature_names=feature_names,
            shap_vector=np.asarray(local_values, dtype=float),
            max_features=max_features,
            title=figure_title,
        )
        LOGGER.info(
            "SHAP explanation generation succeeded for task=%s", task_key)
        error_message = None
        if shap_import_error is not None:
            error_message = f"SHAP dependency is not available: {shap_import_error}"
        return ShapRuntimeResult(success=True, figure=figure, error=error_message)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "SHAP explanation generation failed for task=%s", task_key)
        return ShapRuntimeResult(success=False, error=str(exc))
