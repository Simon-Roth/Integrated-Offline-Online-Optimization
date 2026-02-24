from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from generic.core.config import Config, load_config_data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    merged: dict = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    bgap_path: str | Path,
    *,
    generic_path: str | Path | None = None,
) -> Config:
    """
    Load config by merging a generic YAML with BGAP overrides.
    """
    bgap_path = Path(bgap_path)
    if generic_path is None:
        candidate = bgap_path.parent.parent / "generic" / "generic.yaml"
        if candidate.exists():
            generic_path = candidate
        else:
            generic_path = bgap_path.with_name("generic.yaml")
    generic_path = Path(generic_path)

    generic_data = yaml.safe_load(generic_path.read_text())
    bgap_data = yaml.safe_load(bgap_path.read_text())
    merged = _deep_merge(generic_data, bgap_data)
    return load_config_data(merged)
