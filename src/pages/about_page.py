"""About and methodology page."""

from __future__ import annotations

import streamlit as st

from config.paths import load_manifest, resolve_path
from src.components.disclaimer import render_disclaimer
from src.components.figure_display import display_image_group, display_optional_csv
from src.components.header import render_page_header


def render_about_page() -> None:
    """Render project methodology and artifact summary."""
    manifest = load_manifest()

    render_page_header("About / Methodology",
                       "Research framing and explainability architecture")
    render_disclaimer()

    st.markdown(
        """
        This prototype is based on a research workflow that compares black-box predictive models
        across diabetes, heart disease, and pneumonia tasks and augments outputs with explainability views.
        """
    )

    st.subheader("Four-Layer Framework")
    st.markdown(
        """
        1. Data and preprocessing layer: disease-specific feature preparation and quality controls.
        2. Model layer: trained predictive models reused from research artifacts.
        3. Explainability layer: SHAP, LIME, Grad-CAM, and deterministic text summaries.
        4. Clinical interpretation layer: clinician-oriented presentation and safety warnings.
        """
    )

    st.subheader("Available Disease Modules")
    st.markdown(
        """
        - Diabetes Risk Prediction (stacking model with imputer and scaler artifacts)
        - Heart Disease Prediction (saved sklearn pipeline)
        - Pneumonia Detection (DenseNet121 Keras model)
        """
    )

    st.subheader("Model Artifact Paths")
    for task_key, task_data in manifest["tasks"].items():
        st.write(f"{task_key.title()}: {task_data['model_path']}")

    st.subheader("Explanation Methods")
    st.markdown(
        """
        - Tabular: static SHAP and LIME figures from notebook artifacts, plus rule-based text.
        - Imaging: live Grad-CAM generation with static fallback figures.
        - Confidence bands: configurable threshold mapping from config.
        """
    )

    st.subheader("Cross-Disease Summary Figures")
    dashboard_figure_paths = [
        resolve_path(path_str) for path_str in manifest["dashboard"]["static_figures"]
    ]
    display_image_group(
        dashboard_figure_paths,
        columns=2,
        image_width=560,
        use_container_width=True,
    )

    st.subheader("Performance and XAI Tables")
    dashboard_tables = manifest["dashboard"]["tables"]
    selected_tables = [
        table_path
        for table_path in dashboard_tables
        if table_path.endswith("FINAL_performance_table.csv") or table_path.endswith("FINAL_xai_quality_table.csv")
    ]
    for table_path in selected_tables:
        resolved = resolve_path(table_path)
        display_optional_csv(resolved, title=resolved.name)

    render_disclaimer()
