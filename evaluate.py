# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Run frozen checkpoints on every original validation and test row."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(limits):
        return nullcontext()
try:
    from . import models
    from .metrics import METHOD_ORDER, task_methods
except ImportError:
    import models
    from metrics import METHOD_ORDER, task_methods

ROOT = Path(__file__).resolve().parent


def load_features(archive, bank, stage, limit=None):
    x = np.asarray(bank[archive[stage+'_x'][:limit]], np.float32)
    return dict(x=x, pair_mu=models.pair_features(x, bank[archive[stage+'_mu'][:limit]]),
                pair_pi=models.pair_features(x, bank[archive[stage+'_pi'][:limit]]))


def evaluate_task(task, data_dir, checkpoints, bank, device='cpu'):
    task_id = task['task_id']
    checkpoint = torch.load(checkpoints/(task_id+'.pt'), map_location='cpu', weights_only=False)
    if checkpoint['task_id'] != task_id:
        raise ValueError('Checkpoint task identity differs')
    tables = []
    with np.load(data_dir/'tasks'/(task_id+'.npz'), allow_pickle=False) as archive:
        for stage in ('validation', 'test'):
            features = load_features(archive, bank, stage)
            reward = archive[stage+'_r_mu'].astype(np.float64)
            values = models.predict_values(checkpoint, features, reward, device=device)
            if set(values) != set(task_methods(task['methods'])):
                raise ValueError('Checkpoint methods differ from task metadata')
            n = len(reward)
            if n != int(task['n_'+stage]):
                raise ValueError('Wrong evaluation row count')
            result = {key: [task[key]]*n for key in ('task_id', 'backbone', 'dataset', 'condition', 'data_seed')}
            result.update(prompt_id=archive[stage+'_prompt_ids'].astype(str), source=[stage]*n,
                target_reward=archive[stage+'_r_pi'].astype(np.float64), behavior_reward=reward)
            for method in METHOD_ORDER:
                result[method+'__contribution'] = np.asarray(values[method], np.float64) if method in values else np.full(n, np.nan)
            tables.append(pa.Table.from_pydict(result))
    return pa.concat_tables(tables)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--checkpoints', type=Path, default=ROOT/'trained_checkpoints')
    p.add_argument('--output', type=Path, default=ROOT/'results/predictions.parquet')
    p.add_argument('--task')
    p.add_argument('--backbone', type=str)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    tasks = pd.read_csv(args.data/'tasks.csv').to_dict('records')
    if args.task:
        tasks = [t for t in tasks if t['task_id'] == args.task]
    if args.backbone:
        tasks = [t for t in tasks if t['backbone'] == args.backbone]
    if not tasks:
        p.error('No matching tasks')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp.'+str(os.getpid()))
    writer, banks = None, {}
    summary = []
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):
            for i, task in enumerate(tasks, 1):
                key = task['backbone']+'__'+task['dataset']
                if key not in banks:
                    banks[key] = np.load(args.data/'embeddings'/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                table = evaluate_task(task, args.data, args.checkpoints, banks[key], args.device)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
                for stage, rows in table.to_pandas().groupby('source'):
                    target = float(rows.target_reward.mean())
                    for method in task_methods(task['methods']):
                        estimate = float(rows[method+'__contribution'].mean())
                        summary.append(dict(task_id=task['task_id'], source=stage, method=method,
                                            n_prompts=len(rows), estimate=estimate,
                                            target_value=target, squared_error=(estimate-target)**2))
                print(f'{i}/{len(tasks)} {task["task_id"]}: {len(table)} rows', flush=True)
        writer.close()
        writer = None
        temporary.replace(args.output)
    finally:
        if writer is not None:
            writer.close()
    summary_path = args.output.with_suffix(".summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(args.output)
    print(summary_path)


if __name__ == '__main__':
    main()
