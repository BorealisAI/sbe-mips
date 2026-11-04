# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


def _extract_prompt_from_text(text: str) -> str:
    cleaned = text.strip()
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:", 1)[0].strip()
    return cleaned


def _extract_prompt_from_row(row: dict[str, Any], candidates: list[str]) -> str | None:
    for field in candidates:
        if field in row and row[field] is not None:
            value = row[field]
            if isinstance(value, str) and value.strip():
                return _extract_prompt_from_text(value)
            if isinstance(value, list) and value:
                flattened = "\n".join(str(x) for x in value)
                return _extract_prompt_from_text(flattened)
    # Anthropic HH-style extraction from chosen conversation
    if "chosen" in row and isinstance(row["chosen"], str):
        chosen = row["chosen"].strip()
        if chosen:
            return _extract_prompt_from_text(chosen)
    return None


def _deduplicate_prompts(prompts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for prompt in prompts:
        normalized = re.sub(r"\s+", " ", prompt).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def filter_prompts_by_length(
    prompts: list[str],
    source_tokenizer_id: str,
    max_length: int,
    batch_size: int = 4096,
) -> list[str]:
    """
    Filters prompts based on tokenized length using the specified tokenizer.
    If source_tokenizer_id is empty, returns prompts unfiltered.
    Arguments:
        - prompts: List of prompt strings to filter.
        - source_tokenizer_id: ID of the tokenizer to use for length filtering.
        - max_length: Maximum allowed tokenized length.
        - batch_size: Number of prompts to process at a time.
    Returns:
        - List of prompts that are within the max_length when tokenized.
    """
    if not source_tokenizer_id:
        return prompts
    tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_id, use_fast=True)
    kept = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length + 1,
            return_length=True,
            padding=False,
        )
        lengths = enc["length"]

        for r, l in zip(batch, lengths):
            if l <= max_length:
                kept.append(r)

    return kept


def load_prompts(config: dict[str, Any], logger) -> pd.DataFrame:
    source_cfg = config.get("dataset", {})

    n_total = source_cfg.get("n", -1)

    ds_name = "synthetic"
    source = source_cfg.get("source", "synthetic")
    if source == "synthetic":
        prompts = [
            f"User asks for advice about topic #{i}: explain clearly and provide steps."
            for i in range(n_total * 2)
        ]
    else:
        ds_name = source_cfg.get("hf_name", "Anthropic/hh-rlhf")
        ds_split = source_cfg.get("hf_split", "train")
        ds = load_dataset(ds_name, split=ds_split)

        candidates = source_cfg.get(
            "prompt_field_candidates",
            ["prompt", "question", "instruction", "chosen", "text"],
        )
        prompts = []
        for row in ds:
            prompt = _extract_prompt_from_row(row, candidates)
            if prompt is not None:
                prompts.append(prompt)
        prompts = _deduplicate_prompts(prompts)

    if len(prompts) < n_total:
        raise ValueError(
            f"Insufficient prompts ({len(prompts)}) for requested size {n_total}."
        )

    logger.info("Dropping long prompts based on tokenizer length filtering...")
    prompts_filtered = filter_prompts_by_length(
        prompts,
        source_tokenizer_id=source_cfg.get("source_tokenizer", ""),
        max_length=source_cfg.get("max_prompt_length", 512),
    )

    logger.info(
        f"Loaded {len(prompts)} prompts, {len(prompts_filtered)} after length filtering with tokenizer {source_cfg.get('source_tokenizer', '')} and max length {source_cfg.get('max_prompt_length', 512)}"
    )

    if n_total > 0:
        prompts_filtered = prompts_filtered[:n_total]
    if not prompts_filtered:
        raise ValueError("No prompts remain after filtering")
    records = []
    for i, prompt in enumerate(prompts_filtered):
        records.append(
            {
                "prompt_id": i,
                "prompt": prompt,
                "max_prompt_length": source_cfg.get("max_prompt_length", 512),
                "dataset_key": source_cfg.get("dataset_key", ds_name),
            }
        )

    return pd.DataFrame.from_records(records)
