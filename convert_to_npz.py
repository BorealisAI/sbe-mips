# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Convert scored policy responses into the arrays consumed by train.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    path = Path(path)
    return pd.read_json(path, lines=True) if path.suffix == '.jsonl' else pd.read_parquet(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--behavior', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--dataset', required=True, choices=['hsteer', 'ufb'])
    p.add_argument('--reward', default='skywork', choices=['armorm', 'rm_deberta', 'skywork'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=ROOT/'data')
    p.add_argument('--template', type=Path, default=ROOT/'configs/training.json')
    p.add_argument('--backbone', default='gte')
    p.add_argument('--limit-prompts', type=int, help='Use the first N prompt IDs, retaining all their generations')
    args = p.parse_args()
    if args.seed < 0:
        p.error('seed must be nonnegative')
    mu, pi = read_rows(args.behavior), read_rows(args.target)
    if args.limit_prompts is not None:
        if args.limit_prompts < 12:
            p.error('--limit-prompts must be at least 12')
        ids = sorted(mu.prompt_id.unique())[:args.limit_prompts]
        mu, pi = mu[mu.prompt_id.isin(ids)], pi[pi.prompt_id.isin(ids)]
    reward = 'reward/' + args.reward
    for table in (mu, pi):
        required = {'prompt_id', 'prompt', 'response', 'gen_k', 'model_id', reward}
        if required - set(table):
            raise ValueError(f'Missing columns: {sorted(required-set(table))}')
        if table[list(required)].isna().any().any() or not np.isfinite(table[reward]).all():
            raise ValueError('Missing/nonfinite text, IDs, or rewards; finish reward scoring first')
        if table.duplicated(['prompt_id', 'gen_k']).any() or table.model_id.nunique() != 1:
            raise ValueError('Require unique (prompt_id, gen_k) rows from one policy per input')
        if (table.groupby('prompt_id').prompt.nunique() != 1).any():
            raise ValueError('A prompt ID maps to multiple texts')
    a = mu.groupby('prompt_id').prompt.first().sort_index()
    b = pi.groupby('prompt_id').prompt.first().sort_index()
    if not a.equals(b):
        raise ValueError('Policy prompt IDs/texts differ; align datasets explicitly before preparing')
    if a.duplicated().any():
        raise ValueError('Duplicate prompt texts under different IDs would cause split leakage')
    if len(a) < 12:
        raise ValueError('At least 12 distinct prompts are required')
    # Sample one logged action and one target action per prompt, independently.
    # Ground truth uses ALL available target generations for that prompt.
    rng = np.random.default_rng(args.seed)
    selected = []
    for table in (mu, pi):
        selected.append(table.sort_values(['prompt_id', 'gen_k']).groupby('prompt_id', sort=True)
                        .sample(n=1, random_state=int(rng.integers(2**31))).set_index('prompt_id').loc[a.index])
    left, right = selected
    ids = rng.permutation(a.index.to_numpy())
    ntrain, nvalid = int(.70*len(ids)), max(1, int(.15*len(ids)))
    # Reserve a monitoring partition within the training prompts.
    nearly = max(1, int(.1*ntrain))
    stages = {'fit': ids[:ntrain-nearly], 'early': ids[ntrain-nearly:ntrain],
              'validation': ids[ntrain:ntrain+nvalid], 'test': ids[ntrain+nvalid:]}
    backbone = args.backbone
    transfer = left.model_id.iloc[0].split('/')[-1] + '_to_' + right.model_id.iloc[0].split('/')[-1]
    task = f'{backbone}__{args.dataset}__{args.reward}__{transfer}__seed{args.seed}'
    out = args.output
    (out/'embeddings').mkdir(parents=True, exist_ok=True); (out/'tasks').mkdir(exist_ok=True)
    bank_key = backbone+'__'+args.dataset
    index_path = out/'embeddings'/(bank_key+'.json')
    bank_path = index_path.with_suffix('.npy')
    texts = json.loads(index_path.read_text())
    bank = np.load(bank_path, mmap_mode='r', allow_pickle=False)
    if bank.ndim != 2 or len(bank) != len(texts):
        raise ValueError('Embedding bank and text index differ')
    lookup = {text: i for i, text in enumerate(texts)}
    missing = set([*left.prompt, *left.response, *right.response]) - lookup.keys()
    if missing:
        raise ValueError(f'{len(missing)} texts have no embedding; run calculate_embeddings.py first')
    arrays = {}
    truth = pi.groupby('prompt_id')[reward].mean()
    for stage, prompt_ids in stages.items():
        l, r = left.loc[prompt_ids], right.loc[prompt_ids]
        arrays[stage+'_prompt_ids'] = prompt_ids.astype(str)
        for suffix, values in [('x', l.prompt), ('mu', l.response), ('pi', r.response)]:
            arrays[stage+'_'+suffix] = np.array([lookup[t] for t in values], dtype=np.int64)
        arrays[stage+'_r_mu'] = l[reward].to_numpy(np.float32)
        if stage in ('validation', 'test'):
            arrays[stage+'_r_pi'] = truth.loc[prompt_ids].to_numpy(np.float32)
    np.savez_compressed(out/'tasks'/(task+'.npz'), **arrays)
    cfg = json.loads(args.template.read_text())
    cfg.update(task_id=task, backbone=backbone, dataset=args.dataset, initialization_seed=args.seed)
    configs_path = out/'task_configs.json'
    configs = json.loads(configs_path.read_text()) if configs_path.exists() else {'tasks': {}}
    configs['tasks'][task] = cfg
    configs_path.write_text(json.dumps(configs, indent=2)+'\n')
    metadata = dict(task_id=task, backbone=backbone, dataset=args.dataset, condition=args.reward+'__'+transfer,
                    data_seed=args.seed, reward=args.reward, transfer=transfer,
                    methods='dm;mips;mips_with_reward_supervision;sbe_mips',
                    **{'n_'+stage: len(v) for stage, v in stages.items()})
    csv = out/'tasks.csv'
    rows = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    if len(rows): rows = rows[rows.task_id != task]
    pd.concat([rows, pd.DataFrame([metadata])], ignore_index=True).to_csv(csv, index=False)
    provenance = dict(behavior=str(args.behavior), target=str(args.target),
                      seed=args.seed, reward=args.reward, configuration='task_configs.json',
                      splits={k: v.tolist() for k,v in stages.items()},
                      target_reward='Mean over all available target generations per evaluation prompt')
    (out/'tasks'/(task+'.json')).write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Prepared {task}: '+str({k:len(v) for k,v in stages.items()}))


if __name__ == '__main__':
    main()
