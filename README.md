# Simbid

Simbid is an open-source snapshot of a current-action Decision Transformer pipeline for auction bidding.
It includes training, testing, and algorithm modules.

## Layout

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

The original runs use a conda environment named `Simbid`.

```bash
conda create -n Simbid python=3.9
conda activate Simbid
pip install -r requirements.txt
```

Tested environment (server `Simbid`):
- Python 3.9.25
- PyTorch 2.6.0 (CUDA 12.6)
- transformers 4.57.3
- sentence-transformers 5.1.2
- numpy 2.0.2
- pandas 2.3.1

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
