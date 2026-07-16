"""Safety disclaimer rendering component."""

from __future__ import annotations

import streamlit as st

from config.settings import SAFETY_DISCLAIMER


def render_disclaimer() -> None:
    """Render prominent warning for clinical decision-support scope."""
    st.warning(SAFETY_DISCLAIMER)
