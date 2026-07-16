"""Session state helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st


def init_session_defaults() -> None:
    """Initialize shared session keys once."""
    defaults = {
        "last_task": None,
        "last_prediction": None,
        "last_probability": None,
        "last_confidence_band": None,
        "last_summary": None,
        "assist_context": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_prediction(task: str, label: str, probability: float, confidence_band: str, summary: str) -> None:
    """Persist latest prediction details in session state."""
    st.session_state["last_task"] = task
    st.session_state["last_prediction"] = label
    st.session_state["last_probability"] = probability
    st.session_state["last_confidence_band"] = confidence_band
    st.session_state["last_summary"] = summary


def save_case_context(context: object) -> None:
    """Persist the finished evidence bundle for the online assist panel.

    Stored so the scoped chat survives the reruns that ``st.chat_input`` and the
    suggested-question buttons trigger, without re-running inference. Purely
    additive: when the assist layer is offline nothing reads this back.
    """
    st.session_state["assist_context"] = context
