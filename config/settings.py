"""Application settings pulled from configuration files."""

from __future__ import annotations

from config.paths import get_project_disclaimer, load_manifest, load_thresholds

MANIFEST = load_manifest()
THRESHOLDS = load_thresholds()

APP_TITLE = MANIFEST["project"]["title"]
APP_VERSION = MANIFEST["project"]["version"]
SAFETY_DISCLAIMER = get_project_disclaimer()

CONFIDENCE_HIGH = THRESHOLDS["confidence_bands"]["high"]
CONFIDENCE_MODERATE = THRESHOLDS["confidence_bands"]["moderate"]
