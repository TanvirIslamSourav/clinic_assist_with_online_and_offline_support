"""Streamlit rendering helpers for explainability graph sections."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.components.figure_display import display_optional_image
from src.explainability.shap_runtime import ShapRuntimeResult
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def choose_tabular_render_mode(
    dynamic_available: bool,
    fallback_available: bool,
) -> str:
    """Select rendering strategy for tabular explanation graph section."""
    if dynamic_available:
        return "dynamic"
    if fallback_available:
        return "fallback"
    return "unavailable"


def render_tabular_explanation_graph(
    task_name: str,
    shap_result: ShapRuntimeResult,
    fallback_path: Path | None,
) -> None:
    """Render either dynamic SHAP graph, static fallback, or warning."""
    st.subheader("Feature Contribution Graph")
    st.caption(
        "This graph shows which features influenced the model prediction. "
        "This visualization supports interpretation of model behavior and is not a final diagnosis."
    )

    mode = choose_tabular_render_mode(
        dynamic_available=bool(
            shap_result.success and shap_result.figure is not None),
        fallback_available=fallback_path is not None,
    )

    if mode == "dynamic":
        st.pyplot(shap_result.figure, use_container_width=True)
        plt.close(shap_result.figure)
        LOGGER.info(
            "Displayed dynamic local explanation graph for task=%s", task_name)
        return

    if mode == "fallback":
        st.warning(
            "Prediction completed, but a live local explanation graph could not be generated. "
            "Showing a static fallback figure."
        )
        display_optional_image(
            fallback_path,
            caption=fallback_path.name if fallback_path is not None else None,
            use_container_width=True,
            width=None,
        )
        LOGGER.info(
            "Displayed static explanation fallback image for task=%s", task_name)
        return

    st.warning(
        "Prediction completed successfully, but the explanation graph is unavailable for this input."
    )
    LOGGER.warning(
        "Explanation graph unavailable for task=%s. Dynamic SHAP error=%s",
        task_name,
        shap_result.error,
    )
