"""Orchestrates one scoped Q&A turn: prompt -> Gemini -> validate -> reply.

This is the ONLY entry point the UI uses. It runs strictly downstream of the
model: it consumes a ``CaseContext`` that was built from an already-computed
prediction and returns prose. It cannot produce a probability or a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.client import GeminiClient
from src.llm.config import load_assist_config
from src.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
    suggested_questions,
)
from src.llm.validator import FALLBACK_ANSWER, validate_answer
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Contract cap: 5 turns per case, then the clinician starts a new case.
MAX_TURNS = 5


@dataclass(frozen=True)
class CaseContext:
    """The finished evidence bundle handed to the assist layer.

    Everything here is derived from the saved-artifact prediction. Nothing in it
    is produced by Gemini.
    """

    task_key: str
    task_label: str
    label: str
    probability: float
    confidence_band: str
    positive_contributors: list[str] = field(default_factory=list)
    negative_contributors: list[str] = field(default_factory=list)
    summary: str = ""
    threshold: float = 0.50
    not_seen: str = (
        "the physical examination, patient history, and any labs or imaging not "
        "entered above"
    )

    @property
    def probability_text(self) -> str:
        return f"{self.probability:.2f} ({self.probability * 100:.1f}%)"

    @property
    def threshold_text(self) -> str:
        return f"{self.threshold:.2f}"


@dataclass(frozen=True)
class AssistReply:
    """Result of one turn. ``text`` is always safe to display."""

    text: str
    ok: bool
    reason: str


def get_suggested_questions(context: CaseContext) -> list[str]:
    return suggested_questions(context)


def answer_question(context: CaseContext, question: str) -> AssistReply:
    """Answer one clinician question, grounded in the case context.

    Any offline state, transport failure, empty answer, or validator rejection
    resolves to the fixed fallback line. A clinician never sees a raw failure or
    an unvalidated narrative.
    """
    config = load_assist_config()
    if not config.is_online:
        return AssistReply(FALLBACK_ANSWER, ok=False, reason="offline")

    if not question or not question.strip():
        return AssistReply(FALLBACK_ANSWER, ok=False, reason="empty_question")

    prompt = build_user_prompt(context, question)
    raw = GeminiClient().generate(SYSTEM_INSTRUCTION, prompt)
    if raw is None:
        LOGGER.info("Assist answer unavailable; using fallback.")
        return AssistReply(FALLBACK_ANSWER, ok=False, reason="no_response")

    ok, reason = validate_answer(raw)
    if not ok:
        # Metric, not embarrassment. Log the reason; show the fallback.
        LOGGER.warning("VALIDATOR_REJECT reason=%s", reason)
        return AssistReply(FALLBACK_ANSWER, ok=False, reason=reason)

    return AssistReply(raw.strip(), ok=True, reason="ok")
