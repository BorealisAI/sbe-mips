# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def parse_cli_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        _set_nested(merged, key, value)
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        config = apply_overrides(config, overrides)
    return config
