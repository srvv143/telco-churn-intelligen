"""Configuration loading with single-inheritance (`extends:`) and dict overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config(dict):
    """A plain ``dict`` with a helper that resolves project-relative paths."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config. A file may declare ``extends: other.yaml`` to inherit from it."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = deep_merge(dict(load_config(path.parent / parent)), raw)
    return Config(deep_merge(raw, overrides))
