# Off-Policy Evaluation of Large Language Models via Learned Semantic Bottleneck Embeddings

This repository is the official implementations of "Off-Policy Evaluation of Large Language Models via Learned Semantic Bottleneck Embeddings"(NeurIPS 2026). The following workflow runs an experiment from response generation to the final policy-value estimates.

The example uses **100 HelpSteer2 prompts**, **Llama-3.2-1B-Instruct** as the behavior policy, **Qwen2.5-7B-Instruct** as the target policy, and **Skywork-Reward-V2-Qwen3-8B** as the reward model. Run all commands from the directory containing this README.

## Setup

Use a fresh Python 3.10 environment on a Linux CUDA host for generation and reward scoring:

```bash
conda create -n sbe-mips python=3.10 pip -y
conda activate sbe-mips
python -m pip install --upgrade pip
python -m pip install -r requirements-generation.txt
python -m pip install --no-cache-dir --no-build-isolation --no-deps flash-attn==2.8.3
python -m pip check
python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
```

`requirements-generation.txt` includes vLLM and all training dependencies. It pins vLLM 0.11.0 and PyTorch 2.8.0 so pip resolves them together. Install this file directly; do not first install the training requirements or separately upgrade vLLM/PyTorch. The prebuilt vLLM wheel requires a compatible NVIDIA driver; see the [vLLM installation documentation](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html).

Skywork scoring uses FlashAttention 2. Install `flash-attn==2.8.3` after PyTorch as shown above; `--no-build-isolation` builds against that installation, and `--no-deps` preserves the resolved GPU stack. `--no-cache-dir` avoids cross-device wheel moves when the pip cache is on a different filesystem. Its build and runtime dependencies are included in `requirements-generation.txt`. Source builds require a matching CUDA toolkit with `nvcc`; FlashAttention 2 supports Ampere, Ada, and Hopper GPUs with FP16/BF16. See the [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#installation-and-features).

Model downloads require internet access; gated models require Hugging Face authentication and accepted licenses. For training and evaluation only, install `requirements.txt` in a separate environment; vLLM is not required.

## Models and licenses

| Model | License |
|---|---|
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt) (`llama3.2`) |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache License 2.0 (`apache-2.0`) |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (`gemma`) |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (`gemma`) |
| [Skywork/Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | Apache License 2.0 (`apache-2.0`) |
| [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) | MIT License (`mit`) |
| [RLHFlow/ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) | [Meta Llama 3 Community License](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE) (`llama3`) |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Apache License 2.0 (`apache-2.0`) |

## 1. Generate responses

Generate responses from both policies on the same 100 prompts:

```bash
python generate_data.py \
  --config configs/data_generation/helpsteer_Llama-3.2-1B-Instruct.yaml \
  --run-id hsteer100_llama \
  --set dataset.n=100

python generate_data.py \
  --config configs/data_generation/helpsteer_Qwen2.5-7B-Instruct.yaml \
  --run-id hsteer100_qwen \
  --set dataset.n=100
```

Each policy generates ten responses per prompt, giving **1,000 responses per policy**. Sampling uses temperature 0.7, top-p 0.85, and a maximum of 256 new tokens. Prompts are deduplicated and filtered to 250 tokens under the DeBERTa tokenizer before the first 100 are selected.

The generated datasets are saved to:

```text
data/hsteer100_llama/logged_dataset.parquet
data/hsteer100_qwen/logged_dataset.parquet
```

## 2. Score the responses

Apply Skywork to both generated datasets, using the same configurations and run IDs:

```bash
python score_rewards.py \
  --config configs/data_generation/helpsteer_Llama-3.2-1B-Instruct.yaml \
  --run-id hsteer100_llama

python score_rewards.py \
  --config configs/data_generation/helpsteer_Qwen2.5-7B-Instruct.yaml \
  --run-id hsteer100_qwen
```

Skywork is the only enabled reward model. RM-DeBERTa and ArmoRM remain commented out in the configurations.

The scored files used in the next step are:

```text
results/hsteer100_llama/hsteer_Llama-3.2-1B-Instruct.parquet
results/hsteer100_qwen/hsteer_Qwen2.5-7B-Instruct.parquet
```

Each row contains a prompt, a generated response, its generation index, and the `reward/skywork` score. Existing generated responses can be reused: rerun this scoring stage with the same run IDs, then convert the Skywork rewards into a new prepared-data directory and train new checkpoints.

## 3. Build the training pairs

First calculate the frozen GTE embeddings for the prompts and responses from both policies:

```bash
python calculate_embeddings.py \
  --input results/hsteer100_llama/hsteer_Llama-3.2-1B-Instruct.parquet \
          results/hsteer100_qwen/hsteer_Qwen2.5-7B-Instruct.parquet \
  --model Alibaba-NLP/gte-modernbert-base \
  --dataset hsteer --device cuda --output data/hsteer100
```

This writes a shared embedding bank and its text index under `data/hsteer100/embeddings/`. Prompt and response texts are embedded separately. Change `--model` to another compatible Hugging Face model ID to use a different embedding encoder.

Then match the behavior and target datasets by prompt and convert them into training pairs:

```bash
python convert_to_npz.py \
  --behavior results/hsteer100_llama/hsteer_Llama-3.2-1B-Instruct.parquet \
  --target results/hsteer100_qwen/hsteer_Qwen2.5-7B-Instruct.parquet \
  --dataset hsteer --reward skywork --seed 0 --output data/hsteer100
```

For each prompt, conversion independently selects one behavior response and one target response. The resulting pair contains their embedding indices and the observed behavior reward. All responses belonging to a prompt remain in the same partition.

Conversion uses a 70/15/15 prompt split, with 10% of the training portion reserved for monitoring. The current fixed-epoch trainer does not use monitoring rewards to select epochs. Target rewards are excluded from fitting and monitoring; the evaluation reference averages all ten target rewards for each held-out prompt.

The prepared directory contains:

```text
data/hsteer100/
  embeddings/          # Shared .npy bank and text index
  tasks/               # NPZ training pairs and split metadata
  tasks.csv            # Task names and partition sizes
  task_configs.json    # Training settings for each task
```

Keep these files together. The NPZ arrays reference rows of the shared embedding bank. Training settings are copied from `configs/training.json`; edit that file before conversion, or pass `--template PATH`.

## 4. Train the estimators

```bash
python train.py \
  --data data/hsteer100 \
  --device cuda \
  --output trained_checkpoints/hsteer100
```

This trains DM, MIPS, reward-supervised MIPS, and SBE-MIPS using the prepared pairs. Checkpoints are written to `trained_checkpoints/hsteer100/`. Use `--device cpu` for CPU training.

## 5. Evaluate and read the final results

```bash
python evaluate.py \
  --data data/hsteer100 \
  --checkpoints trained_checkpoints/hsteer100 \
  --output results/hsteer100/predictions.parquet
```

Evaluation runs on the validation and test partitions and produces two files:

| File | Contents |
|---|---|
| `results/hsteer100/predictions.parquet` | Per-prompt estimator contributions and rewards |
| `results/hsteer100/predictions.summary.csv` | Policy-value estimates and squared errors for each method and partition |

Display the final test results in Python:

```python
import pandas as pd
results = pd.read_csv('results/hsteer100/predictions.summary.csv')
print(results.loc[results['source'] == 'test',
                  ['method', 'n_prompts', 'estimate', 'target_value', 'squared_error']]
      .to_string(index=False))
```

`estimate` is the estimated target-policy value. `target_value` is the mean reward of the cached target responses on the same held-out prompts. `squared_error` is their squared difference. Each run reports one squared error per estimator; MSE requires averaging these errors across repeated runs. The 100-prompt example demonstrates the complete workflow on a small dataset.

## Other experiment sizes and policies

For all available prompts, omit `--set dataset.n=100` in both generation commands. Use new run IDs and prepared-data/checkpoint directories when changing the experiment. Unchanged generation and reward-scoring jobs can resume; an existing embedding bank is reused by proceeding directly to conversion rather than recalculating it.

The configuration `configs/data_generation/helpsteer_Gemma3-1b-it.yaml`, provides Gemma-3-1B-it. To use it as either policy, generate and score its responses, include its scored file in embedding calculation, and pass it as `--behavior` or `--target` during conversion.

New embeddings are calculated with mean pooling, L2 normalization, and maximum length 512, then saved directly to a `.npy` bank with a JSON text index. No database is required.

## UltraFeedback generation example

Generate and score 100 UltraFeedback prompts with Llama-3.2-1B-Instruct and Skywork:

```bash
python generate_data.py \
  --config configs/data_generation/ultrafeedback_Llama-3.2-1B-Instruct.yaml \
  --run-id ufb100_llama --set dataset.n=100

python score_rewards.py \
  --config configs/data_generation/ultrafeedback_Llama-3.2-1B-Instruct.yaml \
  --run-id ufb100_llama
```

The scored output is `results/ufb100_llama/ufb_Llama-3.2-1B-Instruct.parquet`. Use `--dataset ufb` for embedding calculation and NPZ conversion. Policy-transfer evaluation also requires responses from a target policy on the same UltraFeedback prompts.

## Included prepared examples

`sample_data/` contains one small HelpSteer2 task for Gemma-3-1B → Gemma-3-12B with Skywork rewards and real cached GTE embeddings. To start directly at training and evaluation:

```bash
python train.py --data sample_data --device cpu --output trained_checkpoints/samples
python evaluate.py --data sample_data --checkpoints trained_checkpoints/samples \
  --output results/samples/predictions.parquet
```

## Reproducibility

Training hyperparameters are defined in `configs/training.json` and recorded for each prepared task in `task_configs.json`. Split IDs and sampling seeds are saved alongside the task archives. The implementation uses product/difference pair features, an additive context baseline, and a representation stage followed by classifier-odds fitting. Target actions are sampled during conversion and remain fixed during training.
