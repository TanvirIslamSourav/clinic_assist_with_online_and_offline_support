"""Isolated Gemini assistance layer.

Everything network-facing lives here. Nothing in this package imports from
``src/pages`` or ``src/components``. The rest of the app talks to it through
``chat_assist`` only, and only ever after the model has already produced its
result. Gemini receives a finished evidence bundle and returns prose; it never
sees, changes, or invents a probability, a class, or a decision.

See ``docs/GEMINI_CONTRACT.md``.
"""

from __future__ import annotations
