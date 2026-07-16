"""Assistance-layer configuration and the online/offline gate.

Reads environment only. The single source of truth for whether the Gemini chat
panel appears at all. If this reports ``offline`` the app behaves exactly as it
did before the assist layer existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Chip states surfaced in the UI. `full` = validated narrative available;
# `offline` = a normal operating state (no key / disabled / no package), not an
# error. There is deliberately no failure state here: a missing key is offline.
STATUS_FULL = "full"
STATUS_OFFLINE = "offline"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AssistConfig:
    """Resolved assist-layer settings. Never carries the key value itself."""

    enabled_flag: bool
    has_key: bool
    package_available: bool
    narration_model: str
    timeout_seconds: float

    @property
    def is_online(self) -> bool:
        """True only when a real Gemini call could succeed.

        All three must hold: the operator turned it on, a key is present, and
        the SDK is installed. Any missing piece degrades silently to offline.
        """
        return self.enabled_flag and self.has_key and self.package_available

    @property
    def status(self) -> str:
        return STATUS_FULL if self.is_online else STATUS_OFFLINE


def _package_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("google.genai") is not None
    except Exception:  # noqa: BLE001 — never let a probe crash the app
        return False


@lru_cache(maxsize=1)
def load_assist_config() -> AssistConfig:
    """Resolve assist config from the environment (cached for the process)."""
    config = AssistConfig(
        enabled_flag=_env_flag("GEMINI_ENABLED", default=False),
        has_key=bool(os.getenv("GEMINI_API_KEY", "").strip()),
        package_available=_package_available(),
        narration_model=os.getenv(
            "GEMINI_NARRATION_MODEL", "gemini-3.5-flash"),
        timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "8")),
    )
    LOGGER.info("Assist layer status resolved: %s", config.status)
    return config


def is_assist_online() -> bool:
    """Convenience gate used by the UI to decide whether to render the panel."""
    return load_assist_config().is_online
