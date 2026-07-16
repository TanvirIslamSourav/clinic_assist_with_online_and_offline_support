"""Home page for the clinician support dashboard."""

from __future__ import annotations

import streamlit as st

from src.components.disclaimer import render_disclaimer
from src.components.header import render_page_header


def render_home_page() -> None:
    """Render home page content."""
    render_page_header(
        title="Explainable AI for Black-Box Models in Medical Diagnosis",
        subtitle="Research-based MVP for clinician decision support",
    )

    render_disclaimer()

    st.markdown(
        """
        This dashboard demonstrates explainable AI techniques applied to three medical prediction tasks.
        It is designed as a decision-support prototype and not as an autonomous diagnosis system.
        """
    )

    st.subheader("Supported Modules")
    st.markdown(
        """
        - Diabetes risk prediction from tabular clinical inputs.
        - Heart disease prediction from tabular clinical inputs.
        - Pneumonia pattern detection from uploaded chest X-ray images.
        """
    )

    st.subheader("Explainability Overview")
    st.markdown(
        """
        - Tabular tasks use rule-based natural-language summaries plus static SHAP/LIME outputs from research artifacts.
        - Image task provides Grad-CAM attention maps when generation succeeds, with static fallback figures if needed.
        - Explanations describe model behavior and should be interpreted with clinical context.
        """
    )
