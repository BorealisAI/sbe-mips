# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


class BaseRewardModel:
    reward_id: str

    def score_batch(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        raise NotImplementedError


class HeuristicRewardModel(BaseRewardModel):
    def __init__(self, reward_id: str) -> None:
        self.reward_id = reward_id

    def score_batch(
        self, prompts: list[str], responses: list[str]
    ) -> np.ndarray:  # noqa: ARG002
        scores = []
        for response in responses:
            n_words = len(response.split())
            civility = (
                1.0
                if any(
                    w in response.lower()
                    for w in ["please", "recommend", "should", "can"]
                )
                else 0.0
            )
            length_score = min(n_words / 40.0, 1.0)
            scores.append(0.7 * length_score + 0.3 * civility)
        return np.asarray(scores, dtype=np.float32), [{} for _ in scores]
