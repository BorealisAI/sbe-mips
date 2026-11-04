# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Calculate a shared text embedding bank for scored policy datasets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True, help='Scored policy Parquet files')
    parser.add_argument('--dataset', choices=['hsteer', 'ufb'], required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    parser.add_argument('--model', default='Alibaba-NLP/gte-modernbert-base')
    parser.add_argument('--backbone', default='gte', help='Name used by convert_to_npz.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit-prompts', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_prompts is not None and args.limit_prompts < 1):
        parser.error('batch-size and limit-prompts must be positive')
    directory = args.output/'embeddings'
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory/(args.backbone+'__'+args.dataset)
    bank_path, index_path = stem.with_suffix('.npy'), stem.with_suffix('.json')
    if bank_path.exists() or index_path.exists():
        raise FileExistsError('Embedding bank already exists; use a new output directory to preserve task indices')

    # Collect each unique text once. Vectors are written in batches to a memory map.
    texts = {}
    for path in args.input:
        allowed = None
        if args.limit_prompts:
            ids = pq.read_table(path, columns=['prompt_id']).column('prompt_id').to_pylist()
            allowed = set(sorted(set(ids))[:args.limit_prompts])
        for batch in pq.ParquetFile(path).iter_batches(columns=['prompt_id', 'prompt', 'response']):
            for row in batch.to_pylist():
                if allowed is not None and row['prompt_id'] not in allowed:
                    continue
                for column in ('prompt', 'response'):
                    text = row[column]
                    if not isinstance(text, str):
                        raise ValueError(f'Non-text {column} in {path}')
                    texts.setdefault(text, None)
    if not texts:
        raise ValueError('No texts found')
    items = list(texts)
    bank = None
    temporary = bank_path.with_suffix('.tmp.npy')
    try:
        from data_generation_pipeline.models.embedder import build_embedder
        encoder = build_embedder({
            'runtime': {'device': args.device, 'dtype': 'float32'},
            'embedding': {'model_id': args.model, 'max_length': 512,
                          'normalize_embeddings': True},
        })
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start+args.batch_size]
            values = encoder.encode_texts(batch, batch_size=args.batch_size)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError('Embeddings must be a finite matrix')
            if bank is None:
                bank = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                                 shape=(len(items), values.shape[1]))
            bank[start:start+len(batch)] = values
        bank.flush()
        del bank
        bank = None
        temporary.replace(bank_path)
        index_path.write_text(json.dumps(list(texts), ensure_ascii=False), encoding='utf-8')
        recipe = 'hashing_md5_v1' if args.model.startswith('dummy/') else 'hf_mean_l2_512_v1'
        stem.with_suffix('.meta.json').write_text(json.dumps(dict(model=args.model,
            recipe=recipe), indent=2)+'\n')
    finally:
        if bank is not None:
            del bank
        if temporary.exists():
            temporary.unlink()
    print(f'{len(items)} embeddings saved to {bank_path}')


if __name__ == '__main__':
    main()
