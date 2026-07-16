"""Sidebar navigation component."""

from __future__ import annotations

import streamlit as st


PAGES = [
    "Home",
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Pneumonia Detection",
    "About / Methodology",
]


def render_sidebar() -> str:
    """Render sidebar and return selected page."""
    with st.sidebar:
        st.header("Navigation")
        selected = st.radio("Select module", options=PAGES, index=0)
        st.markdown("---")
        st.caption("Research MVP for clinician decision support")
    return selected
