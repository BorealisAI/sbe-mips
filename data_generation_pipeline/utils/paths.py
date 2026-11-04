# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str
    data_dir: Path
    results_dir: Path
    logs_dir: Path

    @property
    def logged_dataset(self) -> Path:
        return self.data_dir / "logged_dataset.parquet"

    @property
    def scored_dataset(self) -> Path:
        return self.data_dir / "logged_dataset_scored.parquet"


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_id(config: dict[str, Any], run_id: str | None = None) -> str:
    if run_id:
        return run_id
    configured = config.get("experiment", {}).get("run_id")
    if configured:
        return str(configured)
    return timestamp_run_id()


def prepare_run_paths(root: str | Path, run_id: str) -> RunPaths:
    root_path = Path(root).resolve()
    data_dir = root_path / "data" / run_id
    results_dir = root_path / "results" / run_id
    logs_dir = results_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root_path,
        run_id=run_id,
        data_dir=data_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
    )
