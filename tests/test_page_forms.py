"""Verify manifest-driven forms render widgets and submit successfully."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def _go(page_name: str) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(page_name)
    at.run()
    return at


def test_diabetes_form_submits_cleanly() -> None:
    at = _go("Diabetes Prediction")
    assert not at.exception, at.exception
    # The 'Load sample case' selectbox + 8 input widgets should be present.
    assert at.selectbox, "Expected at least one selectbox on diabetes page"
    # Click the predict button (button with primary type).
    at.button[0].click().run()
    assert not at.exception, f"diabetes predict raised: {at.exception}"


def test_heart_form_submits_cleanly() -> None:
    at = _go("Heart Disease Prediction")
    assert not at.exception, at.exception
    assert at.selectbox, "Expected at least one selectbox on heart page"
    at.button[0].click().run()
    assert not at.exception, f"heart predict raised: {at.exception}"


@pytest.mark.parametrize("sample_label", [
    "Low risk — clear",
    "High risk — clear",
    "Near threshold",
    "Missing values",
])
def test_heart_sample_case_loads(sample_label: str) -> None:
    at = _go("Heart Disease Prediction")
    # First selectbox is 'Load sample case'.
    at.selectbox[0].set_value(sample_label).run()
    assert not at.exception, f"sample load raised: {at.exception}"
    at.button[0].click().run()
    assert not at.exception, f"predict after sample load raised: {at.exception}"


@pytest.mark.parametrize("sample_label", [
    "Low risk — clear",
    "High risk — clear",
    "Near threshold",
    "Missing values",
])
def test_diabetes_sample_case_loads(sample_label: str) -> None:
    at = _go("Diabetes Prediction")
    at.selectbox[0].set_value(sample_label).run()
    assert not at.exception, f"sample load raised: {at.exception}"
    at.button[0].click().run()
    assert not at.exception, f"predict after sample load raised: {at.exception}"
