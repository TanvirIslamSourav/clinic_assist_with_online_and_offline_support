"""Diabetes prediction page."""

from __future__ import annotations

import streamlit as st

from config.paths import get_task_manifest, resolve_path
from src.components.disclaimer import render_disclaimer
from src.components.figure_display import display_image_group
from src.components.gemini_chat import render_assist_panel_for
from src.components.header import render_page_header
from src.components.metric_cards import render_prediction_metrics
from src.explainability.explanation_renderer import render_tabular_explanation_graph
from src.explainability.fallback_loader import get_tabular_fallback_image
from src.explainability.explanation_rules import generate_tabular_explanation
from src.explainability.feature_name_mapping import humanize_feature
from src.explainability.shap_runtime import generate_local_shap_figure
from src.llm.chat_assist import CaseContext
from src.models.diabetes_model_loader import load_diabetes_artifacts
from src.pipelines.diabetes_pipeline import predict_diabetes
from src.utils.field_metadata import FieldSpec, load_field_specs, load_sample_cases
from src.utils.logger import get_logger
from src.utils.session_state import save_case_context, save_prediction

LOGGER = get_logger(__name__)
MODULE = "diabetes"
SPECS = load_field_specs(MODULE)


def _label_for(spec: FieldSpec) -> str:
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


def _apply_sample_case(values: dict) -> None:
    for spec in SPECS:
        if spec.field in values:
            st.session_state[f"diab__{spec.field}"] = values[spec.field]


def _render_field(col, spec: FieldSpec, payloads: dict) -> None:
    """Render one field's widget inside ``col`` and write value to ``payloads``."""
    key = f"diab__{spec.field}"
    default = st.session_state.get(key, spec.default)
    with col:
        if spec.is_categorical:
            index = 0
            for i, opt in enumerate(spec.options):
                if opt.value == default:
                    index = i
                    break
            label_map = {opt.value: opt.label for opt in spec.options}
            value = st.selectbox(
                spec.label,
                options=list(label_map.keys()),
                format_func=lambda v, _m=label_map: _m.get(v, str(v)),
                index=index,
                key=key,
                help=spec.help,
            )
            payloads[spec.field] = value
        elif spec.is_boolean:
            bool_default = bool(default) if default is not None else False
            value = st.checkbox(
                spec.label,
                value=bool_default,
                key=key,
                help=spec.help,
            )
            payloads[spec.field] = value
        else:
            step = spec.step if spec.step is not None else 1.0
            min_v = spec.min if spec.min is not None else 0.0
            max_v = spec.max if spec.max is not None else 1000.0
            fmt = "%d" if spec.type == "integer" else "%g"
            value = st.number_input(
                _label_for(spec),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default if default is not None else min_v),
                step=float(step),
                format=fmt,
                key=key,
                help=spec.help,
            )
            payloads[spec.field] = value


def _split_counts(total: int, groups: int) -> list[int]:
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


def _render_diabetes_form() -> dict:
    """Render diabetes input controls driven by manifest field_specs."""
    sample_cases = load_sample_cases(MODULE)
    sample_options = ["— select —"] + [case.name for case in sample_cases]
    chosen = st.selectbox(
        "Load sample case",
        sample_options,
        index=0,
        key="diab__sample_case",
        help="Fill the form with one of the demo cases.",
    )
    if chosen != "— select —":
        match = next(case for case in sample_cases if case.name == chosen)
        _apply_sample_case(match.values)

    payloads: dict = {}
    columns = st.columns(2)
    counts = _split_counts(len(SPECS), 2)
    cursor = 0
    for col, count in zip(columns, counts):
        for spec in SPECS[cursor : cursor + count]:
            _render_field(col, spec, payloads)
        cursor += count

    return payloads


def render_diabetes_page() -> None:
    """Render diabetes prediction workflow page."""
    render_page_header(
        "Diabetes Prediction",
        "Tabular model inference with rule-based explanation",
    )
    render_disclaimer()

    st.subheader("Patient Inputs")
    payload = _render_diabetes_form()

    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            result = predict_diabetes(payload)
            render_prediction_metrics(
                result.label, result.probability, result.confidence_band
            )

            st.subheader("Local Explanation")
            positive_text = ", ".join(
                humanize_feature(item) for item in result.positive_contributors
            )
            negative_text = ", ".join(
                humanize_feature(item) for item in result.negative_contributors
            )
            st.write(
                f"Key risk-increasing factors: {positive_text or 'Not identified'}"
            )
            st.write(
                f"Potential risk-reducing factors: {negative_text or 'Not identified'}"
            )
            summary = generate_tabular_explanation(
                task_name="Diabetes Risk Prediction",
                predicted_label=result.label,
                predicted_probability=result.probability,
                confidence_band=result.confidence_band,
                positive_contributors=result.positive_contributors,
                negative_contributors=result.negative_contributors,
            )
            st.subheader("Rule-Based Clinical Summary")
            st.info(summary)

            try:
                artifacts = load_diabetes_artifacts()
                shap_result = generate_local_shap_figure(
                    task_key="diabetes",
                    model=artifacts["model"],
                    model_input=result.scaled_features,
                )
                fallback_image = get_tabular_fallback_image("diabetes")
                render_tabular_explanation_graph(
                    task_name="Diabetes Risk Prediction",
                    shap_result=shap_result,
                    fallback_path=fallback_image,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Diabetes explanation rendering failed.")
                st.warning(
                    "Prediction completed successfully, but the explanation "
                    "graph is unavailable for this input."
                )

            save_prediction(
                task="diabetes",
                label=result.label,
                probability=result.probability,
                confidence_band=result.confidence_band,
                summary=summary,
            )
            save_case_context(
                CaseContext(
                    task_key="diabetes",
                    task_label="Diabetes risk",
                    label=result.label,
                    probability=result.probability,
                    confidence_band=result.confidence_band,
                    positive_contributors=[
                        humanize_feature(item)
                        for item in result.positive_contributors
                    ],
                    negative_contributors=[
                        humanize_feature(item)
                        for item in result.negative_contributors
                    ],
                    summary=summary,
                    not_seen=(
                        "the physical examination, patient history, and any "
                        "labs beyond those entered above"
                    ),
                )
            )
            LOGGER.info("Rendered diabetes prediction result.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Diabetes prediction failed.")
            st.error(f"Diabetes prediction failed: {exc}")

    task_manifest = get_task_manifest("diabetes")
    static_figs: list = []
    for key in ("overview", "xai"):
        for path in task_manifest.get("static_figures", {}).get(key, []):
            static_figs.append(resolve_path(path))

    with st.expander("Show static explainability figures", expanded=False):
        if static_figs:
            display_image_group(
                static_figs,
                columns=2,
                image_width=560,
                use_container_width=True,
            )
        else:
            st.info("No static figures configured for this module.")

    render_assist_panel_for("diabetes")
    render_disclaimer()
