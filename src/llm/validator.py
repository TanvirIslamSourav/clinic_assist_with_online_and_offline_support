"""Safety gate on every Gemini answer.

The narration contract's rules 3 and 5: no treatment, no diagnosis language.
This is the one place that decision lives. A rejected answer is discarded and
the UI shows a fixed fallback line instead — a rejected narrative is never shown
to a clinician. Rejections are counted, not hidden.
"""

from __future__ import annotations

import re

# Treatment / prescribing / diagnostic-assertion language. Word-boundaried so
# "mg" does not fire inside "mgmt" and "dose" does not fire inside "diagnose".
_BANNED_PATTERNS: tuple[str, ...] = (
    r"\bprescrib\w*",
    r"\bdose\b",
    r"\bdosage\b",
    r"\bmg\b",
    r"\bml\b",
    r"\bmilligram\w*",
    r"\btablet\w*",
    r"\bmedication\b",
    r"\bdrug\b",
    r"\btreat(?:ment|ed|s)?\b",
    r"\btherapy\b",
    r"\bantibiotic\w*",
    r"\bdiagnos(?:is|e|ed|es|tic)\b",
    r"\bconfirms?\b",
    r"\brule out\b",
    r"\byou should (?:take|start|prescribe)\b",
)

_BANNED_RE = re.compile("|".join(_BANNED_PATTERNS), flags=re.IGNORECASE)

# The single fallback line shown when a model answer is rejected or unavailable.
FALLBACK_ANSWER = (
    "I can't answer that from this case's evidence. I'm limited to explaining "
    "the model's inputs, its probability, and what it did not see. This tool "
    "does not provide diagnoses or treatment guidance."
)


def contains_banned_content(text: str) -> str | None:
    """Return the first banned phrase found, or ``None`` if the text is clean."""
    match = _BANNED_RE.search(text or "")
    return match.group(0) if match else None


def validate_answer(text: str) -> tuple[bool, str]:
    """Validate a model answer.

    Returns ``(ok, reason)``. ``ok`` is False when the answer is empty or
    contains treatment/diagnosis language. The reason is for logging only and is
    never shown to the clinician.
    """
    if not text or not text.strip():
        return False, "empty_response"
    banned = contains_banned_content(text)
    if banned is not None:
        return False, f"banned_content:{banned.lower()}"
    return True, "ok"
