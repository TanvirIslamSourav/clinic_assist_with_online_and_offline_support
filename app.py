"""Streamlit MVP app entrypoint."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import APP_TITLE
from src.components.sidebar import render_sidebar
from src.pages.about_page import render_about_page
from src.pages.diabetes_page import render_diabetes_page
from src.pages.heart_page import render_heart_page
from src.pages.home import render_home_page
from src.pages.pneumonia_page import render_pneumonia_page
from src.utils.logger import get_logger
from src.utils.session_state import init_session_defaults

LOGGER = get_logger(__name__)


def _load_css() -> None:
    css_path = Path(__file__).resolve().parent / \
        "src" / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    """Main app router."""
    st.set_page_config(page_title=APP_TITLE, page_icon="+", layout="wide")
    _load_css()
    init_session_defaults()

    LOGGER.info("Application startup complete.")
    selected_page = render_sidebar()

    if selected_page == "Home":
        render_home_page()
    elif selected_page == "Diabetes Prediction":
        render_diabetes_page()
    elif selected_page == "Heart Disease Prediction":
        render_heart_page()
    elif selected_page == "Pneumonia Detection":
        render_pneumonia_page()
    elif selected_page == "About / Methodology":
        render_about_page()
    else:
        st.error("Unknown page selection.")


if __name__ == "__main__":
    main()
