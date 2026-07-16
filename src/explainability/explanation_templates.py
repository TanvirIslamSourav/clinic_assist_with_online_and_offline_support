"""Template helpers for readable deterministic explanations."""

from __future__ import annotations


def join_phrases(phrases: list[str]) -> str:
    """Join phrases into natural-language comma-separated text."""
    cleaned = [phrase.strip()
               for phrase in phrases if phrase and phrase.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def probability_sentence(probability: float, confidence_band: str) -> str:
    """Return a compact probability sentence."""
    return f"The model output probability is {probability:.1%}, indicating {confidence_band}."
