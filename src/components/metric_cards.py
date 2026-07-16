"""Metric card helpers for prediction result summaries."""

from __future__ import annotations

import streamlit as st


def render_prediction_metrics(label: str, probability: float, confidence_band: str) -> None:
    """Render prediction label, probability, and confidence metrics."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Result", label)
    col2.metric("Probability", f"{probability:.1%}")
    col3.metric("Confidence Band", confidence_band)
