# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
import torch
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(limits):
        return nullcontext()
try:
    from . import models
    from .training import fit_all
    from .evaluate import load_features
except ImportError:
    import models
    from training import fit_all
    from evaluate import load_features

ROOT = Path(__file__).resolve().parent


def file_digest(path):
    """Read the bytes; shared filesystems can preserve timestamps on rewrites."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def validate_resume(bundle, identity):
    models.validate_checkpoint(bundle)
    if bundle.get('task_id') != identity['task_id'] or bundle.get('training') != identity:
        raise ValueError('Existing training output belongs to different data, code, or configuration')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task')
    p.add_argument('--config', type=Path, help='Defaults to DATA/task_configs.json')
    p.add_argument('--backbone', type=str)
    p.add_argument('--device', default='cuda')
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'trained_checkpoints')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--smoke', action='store_true', help='128 fitting/monitoring rows and at most two epochs; separate outputs')
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        p.error('Require 0 <= shard < num-shards')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device unavailable')
    reference = ROOT/'checkpoints'
    if args.output.resolve() == reference.resolve() or args.output.resolve().is_relative_to(reference.resolve()):
        raise ValueError('Training output must be separate from supplied checkpoints')
    config_path = args.config or args.data/'task_configs.json'
    configs = json.loads(config_path.read_text())['tasks']
    names = sorted(configs)
    if args.task:
        if args.task not in configs:
            p.error('Unknown task ID')
        names = [args.task]
    if args.backbone:
        names = [t for t in names if configs[t]['backbone'] == args.backbone]
    if not names:
        p.error('No matching tasks')
    names = names[args.shard::args.num_shards]
    output = args.output/('smoke' if args.smoke else '')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    banks, bank_hashes = {}, {}
    source_hashes = {name: file_digest(ROOT/name) for name in
                     ('train.py', 'training.py', 'models.py', 'odds.py', 'evaluate.py')}
    with threadpool_limits(limits=2):
        for i, task_id in enumerate(names, 1):
            cfg = configs[task_id]
            key = cfg['backbone']+'__'+cfg['dataset']
            bank_path = args.data/'embeddings'/(key+'.npy')
            data_path = args.data/'tasks'/(task_id+'.npz')
            if key not in bank_hashes:
                bank_hashes[key] = file_digest(bank_path)
            signature = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            identity = dict(task_id=task_id, config_sha256=signature, smoke=args.smoke,
                            fixed_hyperparameters=True, fixed_epochs=True, target_rewards_used=False,
                            data_sha256=file_digest(data_path), embedding_bank_sha256=bank_hashes[key],
                            source_sha256=source_hashes)
            path = output/(task_id+'.pt')
            if path.exists():
                bundle = torch.load(path, map_location='cpu', weights_only=False)
                validate_resume(bundle, identity)
                print('ALREADY COMPLETE', task_id, flush=True)
                continue
            if key not in banks:
                banks[key] = np.load(bank_path, mmap_mode='r', allow_pickle=False)
            bank = banks[key]
            arrays = {}
            with np.load(data_path, allow_pickle=False) as archive:
                for stage in ('fit', 'early'):
                    limit = 128 if args.smoke else None
                    arrays[stage] = load_features(archive, bank, stage, limit)
                    arrays[stage]['r_mu'] = archive[stage+'_r_mu'][:limit].astype(np.float32)
            print(f'TRAIN {i}/{len(names)} {task_id} rows={len(arrays["fit"]["r_mu"])} smoke={args.smoke}', flush=True)
            bundle = fit_all(arrays['fit'], arrays['early'], cfg, device=args.device,
                             epoch_cap=2 if args.smoke else None)
            bundle.update(task_id=task_id, training=identity)
            models.validate_checkpoint(bundle)
            temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
            torch.save(bundle, temporary)
            temporary.replace(path)
            print('SAVED', path, flush=True)


if __name__ == '__main__':
    main()
