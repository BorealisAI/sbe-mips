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


class ArmoRMRewardModel(BaseRewardModel):
    def __init__(
        self,
        reward_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.reward_id = reward_id
        self.TOP_K_OBJECTIVES = 3
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_id, use_fast=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_id, trust_remote_code=True, torch_dtype=dtype, device_map=device
        )
        self.model.eval()
        self.device = device

        self.ARMO_ATTRIBUTES = [
            "helpsteer-helpfulness",
            "helpsteer-correctness",
            "helpsteer-coherence",
            "helpsteer-complexity",
            "helpsteer-verbosity",
            "ultrafeedback-overall_score",
            "ultrafeedback-instruction_following",
            "ultrafeedback-truthfulness",
            "ultrafeedback-honesty",
            "ultrafeedback-helpfulness",
            "beavertails-is_safe",
            "prometheus-score",
            "argilla-overall_quality",
            "argilla-judge_lm",
            "code-complexity",
            "code-style",
            "code-explanation",
            "code-instruction-following",
            "code-readability",
        ]

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

        multi_obj_rewards = outputs.rewards.float().cpu()
        gating_output = outputs.gating_output.float().cpu()
        preference_score = outputs.score.float().cpu()

        obj_transform = self.model.reward_transform_matrix.data.float().cpu()
        multi_obj_coeffs = gating_output @ obj_transform.T

        top_obj_dims = torch.argsort(
            torch.abs(multi_obj_coeffs), dim=1, descending=True
        )[:, : self.TOP_K_OBJECTIVES]
        top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

        if "attention_mask" in enc:
            n_tokens = enc["attention_mask"].sum(dim=1).tolist()
            n_tokens = [int(x) for x in n_tokens]
        else:
            n_tokens = [int(enc["input_ids"].shape[1])] * multi_obj_rewards.shape[0]

        scores = []
        metadata = []

        for i in range(multi_obj_rewards.shape[0]):
            obj_names = [self.ARMO_ATTRIBUTES[idx.item()] for idx in top_obj_dims[i]]
            obj_coeffs = [float(x) for x in top_obj_coeffs[i].tolist()]

            scores.append(float(preference_score[i].item()))

            metadata.append(
                {
                    "armo_rewards": [float(x) for x in multi_obj_rewards[i].tolist()],
                    "armo_gating_output": [float(x) for x in gating_output[i].tolist()],
                    "armo_multi_obj_coeffs": [
                        float(x) for x in multi_obj_coeffs[i].tolist()
                    ],
                    "armo_topk_obj_names": obj_names,
                    "armo_topk_obj_coeffs": obj_coeffs,
                    "n_tokens": n_tokens[i],
                }
            )

        return scores, metadata
