"""Centralized path and manifest helpers for the Streamlit app."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"


def resolve_path(path_value: str) -> Path:
    """Resolve a manifest path against the project root."""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    manifest_path = CONFIG_DIR / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    thresholds_path = CONFIG_DIR / "thresholds.json"
    with thresholds_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_task_manifest(task_key: str) -> dict[str, Any]:
    manifest = load_manifest()
    return manifest["tasks"][task_key]


def get_project_disclaimer() -> str:
    return load_manifest()["project"]["safety_disclaimer"]
