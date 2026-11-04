# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from data_generation_pipeline.models.language import BaseTextPolicy, PolicyFactory


def load_behavior_policy(model_id: str, config: dict) -> BaseTextPolicy:
    return PolicyFactory.build(model_id, config)
