#!/usr/bin/env python3
"""
Preprocess trajectory CSV into a PKL with 2048-d SemBid embeddings and
optional embedding lookup table for fast testing.
"""
import argparse
import ast
import importlib
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QwenEncoder2048:
    """Qwen-0.5B encoder with 2048-d output (896 -> 2048 projection)."""

    def __init__(self, output_dim=2048, model_name="Qwen/Qwen2.5-0.5B-Instruct",
                 device=None, max_length=256, seed=42, projection_path=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.output_dim = output_dim
        self.max_length = max_length

        set_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": device}
        )
        self.model.eval()

        self.projection = torch.nn.Linear(896, output_dim).to(device).half()
        if projection_path and os.path.exists(projection_path):
            state = torch.load(projection_path, map_location=device)
            self.projection.load_state_dict(state)
        else:
            torch.nn.init.xavier_uniform_(self.projection.weight)
            if projection_path:
                proj_dir = os.path.dirname(projection_path)
                if proj_dir:
                    os.makedirs(proj_dir, exist_ok=True)
                torch.save(self.projection.state_dict(), projection_path)

    def encode_batch(self, texts):
        """Encode a list of texts to numpy array (float32)."""
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        outputs = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_idx:
            return outputs

        valid_texts = [texts[i] for i in valid_idx]

        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
            embedding = hidden.mean(dim=1)  # [batch, 896]
            embedding = self.projection(embedding.half())  # [batch, 2048]
            embedding = embedding.float().cpu().numpy()

        for idx, emb in zip(valid_idx, embedding):
            outputs[idx] = emb
        return outputs


def load_language_generator(template_module):
    module = importlib.import_module(template_module)
    if not hasattr(module, "BiddingLanguageGeneratorWithTask"):
        raise ValueError(f"Template module missing BiddingLanguageGeneratorWithTask: {template_module}")
    return module.BiddingLanguageGeneratorWithTask()


def parse_state_column(df):
    if 'state' not in df.columns:
        raise ValueError("Input data must include a 'state' column.")
    if isinstance(df['state'].iloc[0], str):
        df['state'] = df['state'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def get_reward_value(row):
    if 'reward_continuous' in row:
        return float(row['reward_continuous'])
    if 'reward' in row:
        return float(row['reward'])
    return 0.0


def build_text_columns(df, generator):
    required_cols = ['advertiserNumber', 'deliveryPeriodIndex', 'timeStepIndex', 'action']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    task_texts = ["" for _ in range(len(df))]
    history_texts = ["" for _ in range(len(df))]
    strategy_texts = ["" for _ in range(len(df))]

    grouped = df.groupby(['advertiserNumber', 'deliveryPeriodIndex'])
    for _, group in grouped:
        group = group.sort_values('timeStepIndex')
        cpa_constraint = group['CPAConstraint'].iloc[0] if 'CPAConstraint' in group.columns else 10.0

        prev_state = None
        prev_action = None
        prev_reward = None

        for idx, row in group.iterrows():
            curr_state = row['state']

            if prev_state is None:
                history_text = ""
                suggested_bid = 25.0
            else:
                history_text = generator.generate_history(prev_state, prev_action, prev_reward, curr_state)
                suggested_bid = float(prev_action) if prev_action is not None else 25.0

            task_text = generator.generate_task_description(cpa_constraint)
            strategy_text = generator.generate_strategy(curr_state, suggested_bid)

            task_texts[idx] = task_text
            history_texts[idx] = history_text
            strategy_texts[idx] = strategy_text

            prev_state = curr_state
            prev_action = row['action'] if 'action' in row else None
            prev_reward = get_reward_value(row)

    df['task_description'] = task_texts
    df['history'] = history_texts
    df['strategy'] = strategy_texts
    return df


def build_lookup(texts, encoder, batch_size):
    lookup = {}
    unique_texts = []
    seen = set()
    for text in texts:
        key = text if text is not None else ""
        if key not in seen:
            seen.add(key)
            unique_texts.append(key)

    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start:start + batch_size]
        embs = encoder.encode_batch(batch)
        for text, emb in zip(batch, embs):
            lookup[text] = emb
    return lookup


def map_embeddings(texts, lookup, emb_dim):
    zero_emb = np.zeros(emb_dim, dtype=np.float32)
    return [lookup.get(text, zero_emb) if text else zero_emb for text in texts]


def main():
    parser = argparse.ArgumentParser(description="Preprocess trajectories into 2048-d embeddings.")
    parser.add_argument('--input_csv', type=str, required=True, help='Input trajectory CSV file')
    parser.add_argument('--output_pkl', type=str, required=True, help='Output PKL with embeddings')
    parser.add_argument('--template_module', type=str, default='sembid_templates_high',
                        help='Template module (sembid_templates_high or sembid_templates_low)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='HuggingFace model name for encoding')
    parser.add_argument('--embedding_dim', type=int, default=2048, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Encoding batch size')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lookup_out', type=str, default=None,
                        help='Optional output path for embedding_lookup pickle')
    parser.add_argument('--projection_out', type=str, default=None,
                        help='Optional output path for projection weights (to reuse later)')
    args = parser.parse_args()

    set_seed(args.seed)

    # Add local paths for template modules.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(current_dir, ".."))
    algo_dir = os.path.join(code_dir, "Algorithms")
    sys.path.insert(0, algo_dir)
    sys.path.insert(0, code_dir)

    generator = load_language_generator(args.template_module)
    encoder = QwenEncoder2048(
        output_dim=args.embedding_dim,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
        projection_path=args.projection_out
    )

    df = pd.read_csv(args.input_csv)
    df = parse_state_column(df)
    df = build_text_columns(df, generator)

    # Build embedding lookups.
    task_lookup = build_lookup(df['task_description'].tolist(), encoder, args.batch_size)
    history_lookup = build_lookup(df['history'].tolist(), encoder, args.batch_size)
    strategy_lookup = build_lookup(df['strategy'].tolist(), encoder, args.batch_size)

    df['task_description_embeddings'] = map_embeddings(df['task_description'], task_lookup, args.embedding_dim)
    df['history_embeddings'] = map_embeddings(df['history'], history_lookup, args.embedding_dim)
    df['strategy_embeddings'] = map_embeddings(df['strategy'], strategy_lookup, args.embedding_dim)

    out_dir = os.path.dirname(args.output_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(df, f, protocol=4)

    if args.lookup_out:
        lookup = {
            'task': task_lookup,
            'history': history_lookup,
            'strategy': strategy_lookup
        }
        lookup_dir = os.path.dirname(args.lookup_out)
        if lookup_dir:
            os.makedirs(lookup_dir, exist_ok=True)
        with open(args.lookup_out, 'wb') as f:
            pickle.dump(lookup, f, protocol=4)


if __name__ == '__main__':
    main()
