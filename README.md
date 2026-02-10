<div align="center">
  <h1>SemBid</h1>
  <p>Current-action Decision Transformer pipeline for auction bidding.</p>
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9-blue">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6.0-orange">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.6-green">
    <img alt="Linux" src="https://img.shields.io/badge/Linux-OK-lightgrey">
    <img alt="Git" src="https://img.shields.io/badge/Git-OK-lightgrey">
  </p>
  <p>
    <code>Offline RL</code>
    <code>Decision Transformer</code>
    <code>Auction Bidding</code>
    <code>Prompt Templates</code>
    <code>Embedding Lookup</code>
  </p>
</div>

## Tech Stack Matrix

<table>
  <tr>
    <td><b>LLM Inference Infra</b><br>Qwen2.5-0.5B embeddings<br>Lookup table for test-time speed</td>
    <td><b>Multi-Agent</b><br>Per-advertiser evaluation agents<br>Offline simulation rollouts</td>
  </tr>
  <tr>
    <td><b>RL Post-Training</b><br>Decision Transformer<br>Current-action prediction</td>
    <td><b>Infra & Tools</b><br>PyTorch + CUDA<br>pandas / numpy</td>
  </tr>
</table>

## Repo Layout

- `code/Algorithms`: model + SemBid template modules
- `code/Training`: training entry points
- `code/Testing`: evaluation entry points
- `code/bidding_train_env`: offline evaluation environment and utilities

## File List

```
Simbid/
  README.md                        # Project overview and usage
  requirements.txt                 # Minimal runtime dependencies
  code/
    Algorithms/
      dt.py                         # Transformer building blocks
      sembid_DT.py                  # SemBid model (current-action DT)
      sembid_utils.py               # Dataset loading and batching
      sembid_templates_high.py      # High-conversion prompt templates
      sembid_templates_low.py       # Low-conversion prompt templates
    Training/
      train.py                      # Training entry point
      preprocess_sembid_embeddings.py # CSV -> PKL + embedding lookup
    Testing/
      test.py                       # Evaluation entry point
    bidding_train_env/
      common/                       # Shared utilities
      offline_eval/                 # Offline evaluation environment
      __init__.py
```

## Environment

Environment versions are listed in `env.txt`.

## Quickstart (Training)

1) Prepare data (choose one):
   - Precomputed PKL: `data/train/trajectory_data_2048.pkl`
   - CSV: `data/train/trajectory.csv` (then run the preprocess step below)

2) Preprocess (CSV -> PKL + optional lookup):

```bash
python code/Training/preprocess_sembid_embeddings.py \
  --input_csv data/train/trajectory.csv \
  --output_pkl data/train/trajectory_data_2048.pkl \
  --lookup_out data/train/embedding_lookup.pkl \
  --template_module sembid_templates_high
```

3) Train:

```bash
python code/Training/train.py \
  --data_file data/train/trajectory_data_2048.pkl \
  --output_dir /path/to/model_dir \
  --num_steps 800000 \
  --save_interval 10000
```

## Data

This repo does not include datasets. Please refer to the AuctionNet project for dataset preparation and sources. Place your data under `data/train` and `data/test` and update paths in commands below:

- Training PKL (precomputed embeddings): `data/train/trajectory_data_2048.pkl`
- Test CSV: `data/test/period-7.csv`

Recommended data layout:

```text
data/
  train/
    trajectory_data_2048.pkl
  test/
    period-7.csv
```

## Data Format

Training PKL (pandas DataFrame) must include at least:
- `deliveryPeriodIndex`, `advertiserNumber`, `timeStepIndex`
- `state` (list/array), `action`, `reward` (or `reward_continuous`)
- `task_description_embeddings`, `history_embeddings`, `strategy_embeddings` (each a 2048-d vector)

Optional but useful:
- `next_state`
- `task_description`, `history`, `strategy` (raw text)

Test CSV must include at least:
- `deliveryPeriodIndex`, `advertiserNumber`, `timeStepIndex`
- `pValue`, `pValueSigma`, `leastWinningCost`
- `budget`, `CPAConstraint`, `advertiserCategoryIndex`

## How To Prepare Training PKL

1) Start from a trajectory CSV with `state`, `action`, and `reward` columns.  
2) For each row, compute `prev_state` and `prev_action` within each trajectory.  
3) Use a template module (`sembid_templates_high` or `sembid_templates_low`) to generate
   Task/History/Strategy text per row.  
4) Encode those texts with Qwen/Qwen2.5-0.5B-Instruct into embeddings (default 2048-d).  
5) Save the DataFrame (including embedding columns) to `trajectory_data_2048.pkl`.

## Preprocess (CSV -> PKL + embedding lookup)

You can use the built-in script to generate the 2048-d embeddings and an optional lookup table:

```bash
python code/Training/preprocess_sembid_embeddings.py \
  --input_csv data/train/trajectory.csv \
  --output_pkl data/train/trajectory_data_2048.pkl \
  --lookup_out data/train/embedding_lookup.pkl \
  --template_module sembid_templates_high
```

The `embedding_lookup.pkl` is a pickle dict with:

```
{
  "task": {text: embedding},
  "history": {text: embedding},
  "strategy": {text: embedding}
}
```

## Train

```bash
python code/Training/train.py \
  --data_file data/train/trajectory_data_2048.pkl \
  --output_dir /path/to/model_dir \
  --num_steps 800000 \
  --save_interval 10000
```

## Entry Points

- Train: `code/Training/train.py`
- Test: `code/Testing/test.py`

## Templates

Two prompt template modules are provided:

- High-conversion: `code/Algorithms/sembid_templates_high.py`
- Low-conversion: `code/Algorithms/sembid_templates_low.py`

## Test (single run)

```bash
python code/Testing/test.py \
  --model_dir /path/to/checkpoint_10000 \
  --test_file data/test/period-7.csv \
  --language_emb_dim 2048 \
  --budget_ratio 1.0 \
  --template_module sembid_templates_high \
  --embedding_lookup data/train/embedding_lookup.pkl
```

## Notes

- GPU required for training/testing.
- Qwen/Qwen2.5-0.5B-Instruct is required for on-the-fly prompt encoding when `--embedding_lookup` is not provided.
- Using `--embedding_lookup` skips on-the-fly Qwen encoding and avoids model download at test time.
- For consistent evaluation, use the same `embedding_lookup` generated during preprocessing.
- Training expects a precomputed embedding PKL (Task/History/Strategy embeddings included).
- Update any absolute paths to your local setup.
