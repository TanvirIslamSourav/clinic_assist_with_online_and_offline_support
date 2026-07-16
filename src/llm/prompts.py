"""Prompt construction for the scoped Q&A panel.

The system instruction is the guard rail; the case context is the *only* factual
material Gemini is given. Both are built here so the rules live in one readable
place and can be pinned by a golden test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle
    from src.llm.chat_assist import CaseContext

# Grounding rules, condensed from docs/GEMINI_CONTRACT.md (features 1 and 4).
SYSTEM_INSTRUCTION = (
    "You are a careful assistant helping a doctor interpret the output of a "
    "clinical risk model. You are given a finished evidence bundle for one "
    "case. Follow every rule:\n"
    "1. Use ONLY the facts in the case context below. Add no clinical facts, "
    "epidemiology, or numbers from your own knowledge.\n"
    "2. Reproduce the probability and threshold verbatim. Never round, restate "
    "differently, or compute a new number.\n"
    "3. Never state or imply a diagnosis. Never name a drug, dose, test "
    "protocol, or treatment. Never advise on management.\n"
    "4. If the result is inconclusive, do not assert a direction; say the "
    "result does not discriminate.\n"
    "5. Always be willing to state what the model did NOT see (physical exam, "
    "history, and labs absent from the inputs).\n"
    "6. If a question cannot be answered from the case context, say so plainly "
    "and do not speculate.\n"
    "7. Answer in plain clinical language, at most 90 words, no reassurance and "
    "no filler. You explain the model; you do not make the decision."
)


def _format_contributors(context: "CaseContext") -> str:
    up = ", ".join(context.positive_contributors) or "none identified"
    down = ", ".join(context.negative_contributors) or "none identified"
    return f"- Pushing risk up: {up}\n- Pulling risk down: {down}"


def build_case_context(context: "CaseContext") -> str:
    """Render the evidence bundle as the grounding block for the prompt."""
    lines = [
        "CASE CONTEXT (the only facts you may use):",
        f"- Module: {context.task_label}",
        f"- Model probability: {context.probability_text}",
        f"- Decision threshold: {context.threshold_text}",
        f"- Model's label for this case: {context.label}",
        f"- Confidence band: {context.confidence_band}",
        _format_contributors(context),
        f"- The model did NOT see: {context.not_seen}",
    ]
    if context.summary:
        lines.append(f"- Rule-based summary already shown: {context.summary}")
    return "\n".join(lines)


def build_user_prompt(context: "CaseContext", question: str) -> str:
    """Combine the grounding block with the clinician's question."""
    return (
        f"{build_case_context(context)}\n\n"
        f"DOCTOR'S QUESTION: {question.strip()}\n\n"
        "Answer using only the case context above."
    )


def suggested_questions(context: "CaseContext") -> list[str]:
    """Starter questions so the panel is never an empty box."""
    return [
        "Which factors moved this result the most?",
        "What did the model not take into account?",
        "How confident is this result, and why?",
        "What would I look at next to interpret this?",
    ]
