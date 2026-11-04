# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

from data_generation_pipeline.models.reward_models.base import BaseRewardModel


class HFSequenceRewardModel(BaseRewardModel):
    def __init__(
        self,
        reward_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        self.reward_id = reward_id

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(reward_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_id, torch_dtype=dtype
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def _prepare_texts(self, prompts: list[str], responses: list[str]) -> list[str]:
        return [
            f"{p}\n{self.tokenizer.sep_token}\n{r}" for p, r in zip(prompts, responses)
        ]

    @torch.no_grad()
    def score_batch(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        texts = self._prepare_texts(prompts, responses)
        reward_n_tokens = [
            len(
                self.tokenizer.tokenize(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=None,
                )
            )
            for text in texts
        ]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)

        if outputs.logits.shape[-1] == 1:
            score = outputs.logits[:, 0]
        else:
            score = outputs.logits[:, -1]
        return score.detach().cpu().float().numpy(), [
            {"ntokens": n} for n in reward_n_tokens
        ]
