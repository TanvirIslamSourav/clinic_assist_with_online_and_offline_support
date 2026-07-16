"""Scoped Gemini Q&A panel — the online assistance UI.

Rendered directly under a result. Absent entirely when offline: the panel does
not appear, no disabled box, no placeholder. When it is present it is visibly
secondary to the model's number and every answer carries its own disclaimer.

This component only calls ``src.llm.chat_assist``; it never touches the SDK.
"""

from __future__ import annotations

import streamlit as st

from src.llm.chat_assist import (
    MAX_TURNS,
    AssistReply,
    CaseContext,
    answer_question,
    get_suggested_questions,
)
from src.llm.config import is_assist_online
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_ASSIST_DISCLAIMER = (
    "AI-generated explanation of the model's output. It does not add to, change, "
    "or override the model's result, and it is not medical advice."
)


def _history_key(task_key: str) -> str:
    return f"gemini_chat_history__{task_key}"


def _case_signature(context: CaseContext) -> str:
    """Stable fingerprint of a case; a new prediction changes it."""
    return (
        f"{context.task_key}|{context.probability:.6f}|{context.label}|"
        f"{','.join(context.positive_contributors)}|"
        f"{','.join(context.negative_contributors)}"
    )


def _get_history(context: CaseContext) -> list[dict[str, str]]:
    """Return the message list for this case, resetting it on a new case.

    History is scoped to a single prediction. When the clinician runs a new
    case the signature changes and the conversation (and its turn cap) resets.
    """
    key = _history_key(context.task_key)
    signature = _case_signature(context)
    store = st.session_state.get(key)
    if not isinstance(store, dict) or store.get("signature") != signature:
        store = {"signature": signature, "messages": []}
        st.session_state[key] = store
    return store["messages"]


def _turns_used(history: list[dict[str, str]]) -> int:
    return sum(1 for message in history if message["role"] == "user")


def _submit_turn(context: CaseContext, question: str) -> None:
    """Run one turn and append both sides to the scoped history."""
    history = _get_history(context)
    if _turns_used(history) >= MAX_TURNS:
        return
    history.append({"role": "user", "content": question})
    reply: AssistReply = answer_question(context, question)
    history.append({"role": "assistant", "content": reply.text})


def render_gemini_chat(context: CaseContext) -> None:
    """Render the scoped chat panel, or nothing at all when offline.

    ``context`` is a finished evidence bundle. This function is the last thing a
    page draws for a result; skipping it leaves the offline app unchanged.
    """
    if not is_assist_online():
        # Offline is a normal state: render nothing. The app runs as it is.
        return

    history = _get_history(context)
    turns_left = MAX_TURNS - _turns_used(history)

    with st.expander("Ask about this result (AI assist)", expanded=False):
        st.caption(
            "🟢 AI assist online · answers are grounded only in this case's "
            "inputs and the model's output."
        )

        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if turns_left <= 0:
            st.info(
                "You've reached the question limit for this case. Run a new "
                "prediction to start a fresh conversation.")
            st.caption(_ASSIST_DISCLAIMER)
            return

        if not history:
            st.markdown("**Suggested questions**")
            for index, question in enumerate(get_suggested_questions(context)):
                if st.button(question, key=f"suggested_{context.task_key}_{index}"):
                    _submit_turn(context, question)
                    st.rerun()

        typed = st.chat_input(
            f"Ask a question about this result ({turns_left} left)…")
        if typed:
            _submit_turn(context, typed)
            st.rerun()

        st.caption(_ASSIST_DISCLAIMER)


def render_assist_panel_for(task_key: str) -> None:
    """Render the chat panel for the most recent case, if it is this page's.

    Reads the persisted ``CaseContext`` so the panel survives chat reruns
    without re-running inference. No-op offline (``render_gemini_chat`` returns
    early) and no-op when the last prediction belongs to another page.
    """
    context = st.session_state.get("assist_context")
    if context is not None and getattr(context, "task_key", None) == task_key:
        render_gemini_chat(context)
