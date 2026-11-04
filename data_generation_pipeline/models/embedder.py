# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BaseEmbedder:
    dim: int

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec
        for token in words:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.stack([self._encode_one(t) for t in texts], axis=0)


class HFTextEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool | None = None,
        max_length: int | None = None,
        normalize_embeddings: bool = True,
        hf_token: str | None = None,
    ) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)

        # Jina v2 models expose an .encode(...) helper and require trust_remote_code=True for it.
        if trust_remote_code is None:
            trust_remote_code = "jina-embeddings-v2" in model_id

        # Token: supports gated models (can also be provided via env var HF_TOKEN)
        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            token=token,
        ).to(device)
        self.model.eval()

        self.device = device
        self.normalize_embeddings = bool(normalize_embeddings)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for {model_id}")
        self.dim = int(hidden_size)

        # jina-embeddings-v2 supports up to 8192 sequence length.
        if max_length is None:
            if "jina-embeddings-v2" in model_id:
                max_length = 8192
            else:
                # fall back to something sensible if tokenizer.model_max_length is huge/unset
                tok_ml = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
                max_length = tok_ml if tok_ml < 100_000 else 512
        self.max_length = int(max_length)

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Preferred path for jina-embeddings-v2: use the model's built-in encode() if available.
        if hasattr(self.model, "encode") and callable(getattr(self.model, "encode")):
            try:
                emb = self.model.encode(texts, batch_size=batch_size, max_length=self.max_length)
            except TypeError:
                # Some versions may not accept batch_size; keep it robust.
                try:
                    emb = self.model.encode(texts, max_length=self.max_length)
                except TypeError:
                    emb = self.model.encode(texts)
            emb = np.asarray(emb, dtype=np.float32)
            return emb

        # Fallback: manual mean pooling + optional L2 normalization (as recommended).
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            encoded = self.tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state  # [B, T, H]
            pooled = self._mean_pool(hidden, encoded["attention_mask"])  # [B, H]

            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

            all_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_vecs, axis=0)


def build_embedder(config: dict[str, Any]) -> BaseEmbedder:
    runtime_cfg = config.get("runtime", {})
    embed_cfg = config.get("embedding", {})
    offline_mode = bool(runtime_cfg.get("offline_mode", False))

    model_id = str(embed_cfg.get("model_id", "dummy/hash"))
    if offline_mode or model_id.startswith("dummy/"):
        return HashingEmbedder(dim=int(embed_cfg.get("dim", 256)))

    # Sensible defaults for jina-embeddings-v2:
    # - trust_remote_code=True enables model.encode(...)
    # - max_length defaults to 8192 for jina v2
    trust_remote_code = bool(embed_cfg.get("trust_remote_code", "jina-embeddings-v2" in model_id))
    default_max_len = 8192 if "jina-embeddings-v2" in model_id else 512

    return HFTextEmbedder(
        model_id=model_id,
        device=str(runtime_cfg.get("device", "cpu")),
        torch_dtype=str(runtime_cfg.get("dtype", "float32")),
        trust_remote_code=trust_remote_code,
        max_length=int(embed_cfg.get("max_length", default_max_len)),
        normalize_embeddings=bool(embed_cfg.get("normalize_embeddings", True)),
        hf_token=(str(embed_cfg["hf_token"]) if "hf_token" in embed_cfg else None),
    )
