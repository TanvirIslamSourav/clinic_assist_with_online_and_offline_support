"""Header component for consistent page title rendering."""

from __future__ import annotations

import streamlit as st


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Render standardized page title section."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
