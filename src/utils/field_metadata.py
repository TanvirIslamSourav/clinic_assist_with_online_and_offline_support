"""Typed loader for the per-module form contract.

Forms are rendered from data, not hand-written. Every input field is described
once under ``modules.<module>`` in ``config/manifest.json`` (see
``docs/FIELD_METADATA.md``); this module turns that JSON into typed objects the
UI and pipelines consume. Adding a unit or fixing a label must never require
touching a page file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.paths import load_manifest

VALID_FIELD_TYPES = frozenset({"number", "integer", "category", "boolean"})


@dataclass(frozen=True)
class FieldOption:
    """A single selectable option for a category or boolean field."""

    value: Any
    label: str


@dataclass(frozen=True)
class FieldSpec:
    """A fully described input field. Rendered from data, never hand-written."""

    field: str
    label: str
    type: str
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    help: str | None = None
    options: tuple[FieldOption, ...] = ()

    @property
    def is_categorical(self) -> bool:
        return self.type == "category"

    @property
    def is_boolean(self) -> bool:
        return self.type == "boolean"

    def option_values(self) -> set[Any]:
        """Set of submit-values the field can take.

        Used to check declared options against the fitted encoder. Boolean
        fields always contribute ``{False, True}`` even if no options are
        listed, matching a ``OneHotEncoder`` fitted on a bool column.
        """
        if self.type == "boolean" and not self.options:
            return {False, True}
        return {option.value for option in self.options}


@dataclass(frozen=True)
class SampleCase:
    """A named preset that fills the form for a demo (see FIELD_METADATA.md)."""

    name: str
    values: dict[str, Any] = field(default_factory=dict)


def _module_block(module: str) -> dict[str, Any]:
    modules = load_manifest().get("modules", {})
    if module not in modules:
        raise KeyError(f"No metadata for module '{module}' in manifest.modules")
    return modules[module]


def load_field_specs(module: str) -> list[FieldSpec]:
    """Return the ordered field specs for a module."""
    specs: list[FieldSpec] = []
    for raw in _module_block(module).get("fields", []):
        options = tuple(
            FieldOption(value=opt["value"], label=opt["label"])
            for opt in raw.get("options", [])
        )
        specs.append(
            FieldSpec(
                field=raw["field"],
                label=raw["label"],
                type=raw["type"],
                unit=raw.get("unit"),
                min=raw.get("min"),
                max=raw.get("max"),
                step=raw.get("step"),
                default=raw.get("default"),
                help=raw.get("help"),
                options=options,
            )
        )
    return specs


def field_order(module: str) -> list[str]:
    """Ordered list of raw field names, matching the model's input order."""
    return [spec.field for spec in load_field_specs(module)]


def load_sample_cases(module: str) -> list[SampleCase]:
    """Return the demo presets declared for a module."""
    return [
        SampleCase(name=case["name"], values=dict(case["values"]))
        for case in _module_block(module).get("sample_cases", [])
    ]


def get_population_caveat(module: str) -> str:
    """Return the module's population-limitation string (may be empty)."""
    return _module_block(module).get("population_caveat", "")
