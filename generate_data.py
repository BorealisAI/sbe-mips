# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import math
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation_pipeline.dataio.prompt_source import load_prompts
from data_generation_pipeline.models.behavior import load_behavior_policy
from data_generation_pipeline.models.language import GenerationConfig
from data_generation_pipeline.utils.config import load_config, parse_cli_overrides
from data_generation_pipeline.utils.io import read_parquet_if_exists, write_parquet
from data_generation_pipeline.utils.logging_utils import init_logger
from data_generation_pipeline.utils.paths import prepare_run_paths, resolve_run_id
from data_generation_pipeline.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build logged bandit dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--set", nargs="*", default=None, help="key=value overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    cfg = load_config(root / args.config, parse_cli_overrides(args.set))
    run_id = resolve_run_id(cfg, args.run_id)
    paths = prepare_run_paths(root, run_id)
    logger = init_logger(paths.logs_dir / "01_build_logged_dataset.log")

    seed = int(cfg.get("runtime", {}).get("seed", 13))
    set_global_seed(seed)

    logger.info("Loading prompts...")
    prompts_df = load_prompts(cfg, logger)
    out_path = paths.logged_dataset
    existing = read_parquet_if_exists(out_path)
    k = int(cfg.get("generation", {}).get("k", 1))
    if k < 1:
        raise ValueError("generation.k must be positive")
    done_ids = set()
    if not existing.empty:
        done_ids = {pid for pid, group in existing.groupby("prompt_id")
                    if set(group.gen_k) >= set(range(k))}

    model_id = cfg.get("models", {}).get("language_model", None)
    if not model_id:
        raise ValueError("Config models.language_model is required.")
    model = load_behavior_policy(model_id, cfg)

    gen_cfg_root = cfg.get("generation", {})
    base_gen_cfg = GenerationConfig(
        temperature=float(gen_cfg_root.get("temperature", 0.7)),
        top_p=float(gen_cfg_root.get("top_p", 0.85)),
        max_new_tokens=int(gen_cfg_root.get("max_new_tokens", 256)),
        k=int(gen_cfg_root.get("k", 1)),
        seed=seed,
    )

    new_rows: list[dict] = []
    flush_every = int(cfg.get("runtime", {}).get("flush_every", 100))

    def flush_rows(rows: list[dict]) -> None:
        if not rows:
            return
        nonlocal existing
        new_df = pd.DataFrame.from_records(rows)
        combined = pd.concat([existing, new_df], axis=0, ignore_index=True)
        combined = combined.drop_duplicates(subset=["prompt_id", "gen_k"], keep="last")
        combined = combined.sort_values("prompt_id").reset_index(drop=True)
        write_parquet(combined, out_path)
        existing = combined
        rows.clear()

    logger.info("Generating responses with resume support...")
    batch_size = int(cfg.get("runtime", {}).get("language_batch_size", 1))
    batch_count = math.ceil(len(prompts_df) / batch_size)

    for batch_idx in tqdm(range(batch_count), desc="Building Dataset"):
        start = batch_idx * batch_size
        batch_df = prompts_df.iloc[start : start + batch_size]

        batch_rows = []
        for r in batch_df.itertuples(index=False):
            pid = int(r.prompt_id)
            if pid in done_ids:
                continue
            batch_rows.append(r)

        if not batch_rows:
            continue

        batch_gen_cfg = GenerationConfig(
            temperature=base_gen_cfg.temperature,
            top_p=base_gen_cfg.top_p,
            max_new_tokens=base_gen_cfg.max_new_tokens,
            k=base_gen_cfg.k,
            seed=seed + batch_idx,
        )

        prompts = [r.prompt for r in batch_rows]
        responses = model.generate_batch(prompts, batch_gen_cfg)

        prompts = [
            p for p in prompts for _ in range(base_gen_cfg.k)
        ]  # repeat prompts k times
        batch_rows = [
            r for r in batch_rows for _ in range(base_gen_cfg.k)
        ]  # repeat rows k times
        logprobs_batchified = model.logprob_batch(prompts, responses)

        if len(responses) != len(batch_rows) or len(logprobs_batchified) != len(
            batch_rows
        ):
            raise RuntimeError(
                f"Batch size mismatch: rows={len(batch_rows)} "
                f"responses={len(responses)} logprobs={len(logprobs_batchified)}"
            )

        for idx, (r, response, log_probs_out) in enumerate(
            zip(batch_rows, responses, logprobs_batchified)
        ):
            logp_mu, n_tokens = log_probs_out
            pid = int(r.prompt_id)
            k_idx = idx % base_gen_cfg.k
            new_rows.append(
                {
                    "prompt_id": pid,
                    "gen_k": k_idx,
                    "prompt": r.prompt,
                    "max_prompt_length": int(r.max_prompt_length),
                    "model_id": model_id,
                    "response": response,
                    "logp_mu": float(logp_mu),
                    "n_tokens": int(n_tokens),
                    "gen_temperature": float(batch_gen_cfg.temperature),
                    "gen_top_p": float(batch_gen_cfg.top_p),
                    "max_new_tokens": int(batch_gen_cfg.max_new_tokens),
                    "seed": int(batch_gen_cfg.seed),
                }
            )
            done_ids.add(pid)

        if len(new_rows) >= flush_every:
            flush_rows(new_rows)

    flush_rows(new_rows)
    logger.info("Finished. Rows=%d saved to %s", len(existing), out_path)


if __name__ == "__main__":
    main()
