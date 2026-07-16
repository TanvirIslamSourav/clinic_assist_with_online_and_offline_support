"""Smoke test: drive the Streamlit app through each page route."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

PAGES = [
    "Home",
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Pneumonia Detection",
    "About / Methodology",
]


def _visit(page_name: str) -> None:
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(page_name)
    at.run()
    assert not at.exception, f"{page_name} raised: {at.exception}"


def test_home_route() -> None:
    _visit("Home")


def test_diabetes_route() -> None:
    _visit("Diabetes Prediction")


def test_heart_route() -> None:
    _visit("Heart Disease Prediction")


def test_pneumonia_route() -> None:
    _visit("Pneumonia Detection")


def test_about_route() -> None:
    _visit("About / Methodology")
