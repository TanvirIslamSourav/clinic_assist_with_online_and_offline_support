"""File and artifact utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def ensure_file_exists(path: Path, description: str) -> Path:
    """Validate that a required file exists and raise a clear error if not."""
    if not path.exists() or not path.is_file():
        message = f"Missing required {description}: {path}"
        LOGGER.error(message)
        raise FileNotFoundError(message)
    return path


def optional_existing_paths(paths: Iterable[Path]) -> list[Path]:
    """Return only paths that exist without raising an exception."""
    return [path for path in paths if path.exists() and path.is_file()]


def safe_read_bytes(path: Path) -> bytes:
    """Read file bytes with clear logging around failures."""
    try:
        return path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed reading bytes from %s", path)
        raise IOError(f"Failed to read file: {path}") from exc
