# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from data_generation_pipeline.models.reward_models.base import (
    BaseRewardModel,
    HeuristicRewardModel,
)
from tqdm import tqdm


@dataclass
class RewardModelSpec:
    name: str
    model_id: str


class RewardModelWrapper:
    def __init__(self, spec: RewardModelSpec, config: dict[str, Any]) -> None:
        runtime_cfg = config.get("runtime", {})
        offline_mode = bool(runtime_cfg.get("offline_mode", False))

        self.spec = spec
        if offline_mode or spec.model_id.startswith("dummy/"):
            self.model: BaseRewardModel = HeuristicRewardModel(spec.model_id)
        elif spec.name.startswith("armorm"):
            from .reward_models.armo_rm import ArmoRMRewardModel
            self.model: BaseRewardModel = ArmoRMRewardModel(
                spec.model_id,
                device=str(runtime_cfg.get("device", "cpu")),
                torch_dtype=str(runtime_cfg.get("dtype", "bfloat16")),
            )
        elif spec.name.startswith("skywork"):
            from .reward_models.skywork import SkyworkRewardModel
            self.model: BaseRewardModel = SkyworkRewardModel(
                spec.model_id,
                device=str(runtime_cfg.get("device", "cpu")),
                torch_dtype=str(runtime_cfg.get("dtype", "bfloat16")),
            )
        else:
            from .reward_models.hf_rm import HFSequenceRewardModel
            self.model = HFSequenceRewardModel(
                spec.model_id,
                device=str(runtime_cfg.get("device", "cpu")),
                torch_dtype=str(runtime_cfg.get("dtype", "float32")),
            )

    @property
    def column_name(self) -> str:
        return f"reward/{self.spec.name}"

    def score_in_batches(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        all_scores: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for start in tqdm(range(0, len(prompts), batch_size)):
            end = start + batch_size
            batch_scores, batch_metadata = self.model.score_batch(
                prompts[start:end], responses[start:end]
            )
            all_scores.append(np.asarray(batch_scores, dtype=np.float32).reshape(-1))
            metadata.extend(batch_metadata)
        if not all_scores:
            return np.empty((0,), dtype=np.float32), []
        return np.concatenate(all_scores, axis=0), metadata
