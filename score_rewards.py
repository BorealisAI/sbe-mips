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
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation_pipeline.models.reward import RewardModelSpec, RewardModelWrapper
from data_generation_pipeline.utils.config import load_config, parse_cli_overrides
from data_generation_pipeline.utils.io import read_parquet_if_exists, write_parquet
from data_generation_pipeline.utils.logging_utils import init_logger
from data_generation_pipeline.utils.paths import prepare_run_paths, resolve_run_id
from data_generation_pipeline.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score logged data with reward models")
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
    logger = init_logger(paths.logs_dir / "02_score_rewards.log")

    seed = int(cfg.get("runtime", {}).get("seed", 13))
    set_global_seed(seed)

    logged = read_parquet_if_exists(paths.logged_dataset)
    if logged.empty:
        raise FileNotFoundError(f"Logged dataset not found: {paths.logged_dataset}")

    scored = read_parquet_if_exists(paths.scored_dataset)
    if scored.empty:
        scored = logged.copy()
    else:
        keys = ["prompt_id", "gen_k", "prompt", "response", "model_id"]
        reward_columns = [c for c in scored if c.startswith("reward/")]
        scored = logged.merge(scored[keys + reward_columns], on=keys, how="left", validate="one_to_one")

    reward_specs_cfg = list(cfg.get("models", {}).get("reward_models", []))
    if not reward_specs_cfg:
        raise ValueError("Config models.reward_models is required.")

    reward_specs = [
        RewardModelSpec(name=str(x["name"]), model_id=str(x["model_id"]))
        for x in reward_specs_cfg
    ]
    batch_size = int(cfg.get("runtime", {}).get("reward_batch_size", 16))

    for spec in reward_specs:
        col = f"reward/{spec.name}"

        if col not in scored.columns:
            scored[col] = np.nan

        missing_mask = scored[col].isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            logger.info("Skipping %s: already scored", col)
            continue

        logger.info("Scoring %s for %d rows...", col, n_missing)
        wrapper = RewardModelWrapper(spec, cfg)
        prompts = scored.loc[missing_mask, "prompt"].tolist()
        responses = scored.loc[missing_mask, "response"].tolist()
        scores, metadata = wrapper.score_in_batches(
            prompts, responses, batch_size=batch_size
        )

        target_idx = scored.index[missing_mask]

        for key in metadata[0].keys():
            values = [m[key] for m in metadata]
            scored.loc[missing_mask, f"{col}_{key}"] = pd.Series(
                values, index=target_idx, dtype="object"
            )
        scored.loc[missing_mask, col] = scores.astype(np.float32)

        write_parquet(scored, paths.scored_dataset)
        del wrapper
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored["reward_model_ids"] = ",".join([spec.name for spec in reward_specs])
    scored["reward_scoring_batch_size"] = batch_size
    scored["reward_dtype"] = str(cfg.get("runtime", {}).get("dtype", "float32"))

    model_id = scored["model_id"].iloc[0] if "model_id" in scored.columns else "unknown"
    model_id = model_id.split("/")[-1] if "/" in model_id else model_id
    dataset_key = cfg.get("dataset", {}).get("dataset_key", "unknown")
    results_data_path = paths.results_dir / f"{dataset_key}_{model_id}.parquet"

    write_parquet(scored, paths.scored_dataset)

    # for final use, we save the scored dataset with a more descriptive name in the results directory
    write_parquet(scored, results_data_path)

    logger.info("Finished. Scored dataset saved to %s", paths.scored_dataset)


if __name__ == "__main__":
    main()
