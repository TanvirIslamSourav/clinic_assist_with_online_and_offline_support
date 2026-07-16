"""Thin, defensive wrapper around the google-genai SDK.

Every failure mode — no key, no package, timeout, rate limit, transport error —
returns ``None``. The caller falls back. This module never raises to the UI and
never logs prompt contents at INFO.
"""

from __future__ import annotations

import concurrent.futures
import os

from src.llm.config import load_assist_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GeminiClient:
    """Single-shot text generation. Constructed lazily so import never fails."""

    def __init__(self) -> None:
        self._config = load_assist_config()
        self._client = None

    @property
    def available(self) -> bool:
        return self._config.is_online

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.available:
            return False
        try:
            from google import genai  # imported lazily; may be absent offline

            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            return True
        except Exception:  # noqa: BLE001
            LOGGER.exception("Gemini client initialisation failed.")
            self._client = None
            return False

    def generate(self, system_instruction: str, user_prompt: str) -> str | None:
        """Return model text, or ``None`` on any failure or when offline.

        A hard timeout wraps the call so a stalled connection can never freeze a
        rerun. Temperature is pinned to 0 for reproducible, low-imagination
        answers — this layer explains, it does not brainstorm.
        """
        if not self._ensure_client():
            return None

        try:
            from google.genai import types

            def _call() -> str | None:
                response = self._client.models.generate_content(
                    model=self._config.narration_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                )
                text = getattr(response, "text", None)
                return text.strip() if isinstance(text, str) else None

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                return future.result(timeout=self._config.timeout_seconds)
        except concurrent.futures.TimeoutError:
            LOGGER.warning(
                "Gemini call timed out after %ss; falling back.",
                self._config.timeout_seconds,
            )
            return None
        except Exception:  # noqa: BLE001
            LOGGER.exception("Gemini call failed; falling back.")
            return None
