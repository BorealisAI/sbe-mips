# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

from data_generation_pipeline.models.reward_models.base import BaseRewardModel


class SkyworkRewardModel(BaseRewardModel):
    def __init__(
        self,
        reward_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.reward_id = reward_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.model.eval()
        self.device = device

    def _prepare_texts(self, prompts: list[str], responses: list[str]) -> list[str]:
        chats = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(prompts, responses)
        ]
        formatted = [
            self.tokenizer.apply_chat_template(conversation, tokenize=False)
            for conversation in chats
        ]

        if self.tokenizer.bos_token is not None and formatted[0].startswith(
            self.tokenizer.bos_token
        ):
            formatted = [f[len(self.tokenizer.bos_token) :] for f in formatted]

        return formatted

    @torch.no_grad()
    def score_batch(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        chats = self._prepare_texts(prompts, responses)
        enc = self.tokenizer(chats, return_tensors="pt", padding=True)

        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        scores = outputs.logits[:, 0].float().cpu().tolist()

        if "attention_mask" in enc:
            n_tokens = enc["attention_mask"].sum(dim=1).tolist()
            n_tokens = [int(x) for x in n_tokens]
        else:
            n_tokens = [int(enc["input_ids"].shape[1])] * len(scores)

        return scores, [{"n_tokens": n} for n in n_tokens]
