"""Tests for explanation rendering mode selection."""

from __future__ import annotations

from src.explainability.explanation_renderer import choose_tabular_render_mode


def test_render_mode_prefers_dynamic_when_available() -> None:
    assert choose_tabular_render_mode(True, True) == "dynamic"


def test_render_mode_uses_fallback_when_dynamic_is_missing() -> None:
    assert choose_tabular_render_mode(False, True) == "fallback"


def test_render_mode_marks_unavailable_without_any_graph() -> None:
    assert choose_tabular_render_mode(False, False) == "unavailable"
