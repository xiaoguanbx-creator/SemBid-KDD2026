#!/usr/bin/env python3
"""
Standard test script (GAS/GAVE-style budget handling).
Qwen-0.5B + 2048-d LanguageDT test (History/Strategy before State).
"""
import sys
import torch
import numpy as np
import math
import logging
import json
import pickle
import time
import argparse
import os
import importlib

# Add local paths for project modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.abspath(os.path.join(current_dir, ".."))
algo_dir = os.path.join(code_dir, "Algorithms")
sys.path.insert(0, algo_dir)
sys.path.insert(0, code_dir)

# Import environment and data loader.
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

# Import model and encoder.
from sembid_DT import LanguageGuidedDTWithTaskFlexible

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class QwenEncoder2048:
    """Qwen-0.5B encoder with 2048-d output."""

    def __init__(self, output_dim=2048, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", max_length=256):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device
        self.output_dim = output_dim
        self.max_length = max_length

        logger.info(f"Initializing Qwen-0.5B encoder ({output_dim}-d)...")
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

        # Project native 896-d features to 2048-d.
        self.projection = torch.nn.Linear(896, output_dim).to(device).half()
        torch.nn.init.xavier_uniform_(self.projection.weight)
        logger.info(f"Qwen encoder ready (896 -> {output_dim})")

    def encode(self, text, convert_to_tensor=True, device=None):
        """Encode a single text string."""
        if device is None:
            device = self.device

        if not text or text.strip() == "":
            if convert_to_tensor:
                return torch.zeros(self.output_dim, dtype=torch.float32, device=device)
            return np.zeros(self.output_dim, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # [896]
            embedding = self.projection(embedding.half())  # [2048]

        if convert_to_tensor:
            return embedding.float().to(device)
        return embedding.float().cpu().numpy()


class BiddingStrategy:
    """Qwen-0.5B + 2048-d strategy (standard)."""

    def __init__(self, model_dir, budget, cpa, category, device='cuda',
                 shared_encoder=None, language_emb_dim=2048, embedding_lookup=None,
                 language_generator_cls=None):
        self.budget = budget
        self.remaining_budget = budget
        self.cpa = cpa
        self.category = category
        self.device = device
        self.model_dir = model_dir
        self.language_emb_dim = language_emb_dim
        self.embedding_lookup = embedding_lookup
        self.use_precomputed = embedding_lookup is not None
        # Inference-time knobs (via environment variables).
        self.action_clip = float(os.getenv("SIMBID_ACTION_CLIP", "300"))
        self.action_scale = float(os.getenv("SIMBID_ACTION_SCALE", "1.0"))
        self.rtg_mode = os.getenv("SIMBID_RTG_MODE", "fixed")  # fixed / budget_cpa
        self.rtg_value = float(os.getenv("SIMBID_RTG_VALUE", "8"))
        # Evaluation-time stats for 16-d state construction.
        self.bid_mean_hist = []
        self.conv_mean_hist = []
        self.precomputed_stats = None

        # Load config.
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'use_language': True,
                'language_type': 'both',
                'state_dim': 16,
                'act_dim': 1,
                'max_length': 20,
                'max_ep_len': 96,
                'language_emb_dim': language_emb_dim
            }

        # Load normalization stats.
        norm_path = os.path.join(model_dir, 'normalize_dict.pkl')
        if os.path.isdir(norm_path):
            norm_path = os.path.join(norm_path, 'normalize_dict.pkl')
        if os.path.exists(norm_path) and os.path.isfile(norm_path):
            with open(norm_path, 'rb') as f:
                normalize_dict = pickle.load(f)
            self.state_mean = normalize_dict['state_mean']
            self.state_std = normalize_dict['state_std']
        else:
            logger.warning(f"Normalize dict not found, using zeros")
            self.state_mean = np.zeros(self.config['state_dim'])
            self.state_std = np.ones(self.config['state_dim'])

        # Initialize model.
        self.model = LanguageGuidedDTWithTaskFlexible(
            state_dim=self.config['state_dim'],
            act_dim=self.config['act_dim'],
            state_mean=self.state_mean,
            state_std=self.state_std,
            use_language=self.config['use_language'],
            K=self.config['max_length'],
            max_ep_len=self.config['max_ep_len'],
            language_emb_dim=language_emb_dim,
            use_precomputed_embeddings=True
        )

        # Load model weights.
        model_path = os.path.join(model_dir, 'language_dt.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'language_dt_2048.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'pytorch_model.bin')

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        # Qwen encoder (only needed without lookup table).
        if self.use_precomputed:
            self.encoder = None
            logger.info("Using precomputed embedding lookup (2048-d)")
        elif shared_encoder:
            self.encoder = shared_encoder
            logger.info("Using shared Qwen-0.5B encoder (2048-d)")
        else:
            self.encoder = QwenEncoder2048(output_dim=language_emb_dim, device=device)

        # Language generator (for text used in encoding or lookup).
        if language_generator_cls is None:
            raise ValueError("language_generator_cls is required.")
        self.language_generator = language_generator_cls()

        # History buffers.
        self.max_length = self.config['max_length']
        self.reset_buffer()
        self.emb_cache = {}

    def reset_buffer(self):
        """Reset history buffers."""
        self.states = []
        self.actions = []
        self.timesteps = []
        self.lang_task_desc = []
        self.lang_history = []
        self.lang_strategy = []
        self.last_state = None
        self.last_action = None
        self.last_reward = 0

    def _encode_text_cached(self, text, text_type='task'):
        """Encode text with a small cache."""
        if not text or text.strip() == "":
            return torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)

        if text in self.emb_cache:
            return self.emb_cache[text]

        if len(self.emb_cache) > 1000:
            self.emb_cache.clear()

        # Use lookup table or on-the-fly encoding.
        if self.use_precomputed:
            lookup = self.embedding_lookup.get(text_type, {})
            if text in lookup:
                emb = torch.from_numpy(lookup[text]).float().to(self.device)
            else:
                emb = torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)
        else:
            emb = self.encoder.encode(text, convert_to_tensor=True, device=self.device)

        self.emb_cache[text] = emb
        return emb

    def _get_language_embeddings(self, current_state, suggested_bid=1.0):
        """Generate language embeddings."""
        # Task Description
        task_text = self.language_generator.generate_task_description(self.cpa)
        lang_task_desc_emb = self._encode_text_cached(task_text, text_type='task')

        # History
        lang_history_emb = torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)
        if self.last_state is not None:
            history_text = self.language_generator.generate_history(
                prev_state=self.last_state,
                prev_action=self.last_action,
                reward=self.last_reward,
                current_state=current_state
            )
            if history_text:
                lang_history_emb = self._encode_text_cached(history_text, text_type='history')

        # Strategy
        strategy_text = self.language_generator.generate_strategy(current_state, suggested_bid=suggested_bid)
        lang_strategy_emb = self._encode_text_cached(strategy_text, text_type='strategy') if strategy_text else torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)

        return lang_task_desc_emb, lang_history_emb, lang_strategy_emb

    def bidding(self, timeStep_index, pValue, pValueSigma, historyBids,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """Generate a bid for the current step."""
        # Build state (prefer the training-aligned 16-d features).
        if self.precomputed_stats is not None and self.config.get('state_dim', 16) == 16:
            stats = self.precomputed_stats
            t = int(timeStep_index)
            timeStepIndexNum = 48
            timeleft = (timeStepIndexNum - t) / timeStepIndexNum
            bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0.0

            avg_bid_all = float(np.mean(self.bid_mean_hist)) if self.bid_mean_hist else 0.0
            avg_bid_last_3 = float(np.mean(self.bid_mean_hist[-3:])) if self.bid_mean_hist else 0.0

            pvalue_mean = stats["pvalue_mean"]
            lwc_mean = stats["lwc_mean"]
            xi_mean = stats["xi_mean"]
            volume = stats["volume"]
            hist_volume = stats["historical_volume"]
            last3_volume = stats["last3_volume"]

            def _mean_slice(arr, end, k=3):
                if end <= 0:
                    return 0.0
                start = max(0, end - k)
                return float(np.mean(arr[start:end]))

            avg_lwc_all = float(np.mean(lwc_mean[:t])) if t > 0 else 0.0
            avg_lwc_last_3 = _mean_slice(lwc_mean, t, 3)
            avg_pvalue_all = float(np.mean(pvalue_mean[:t])) if t > 0 else 0.0
            avg_pvalue_last_3 = _mean_slice(pvalue_mean, t, 3)
            avg_conv_all = float(np.mean(self.conv_mean_hist)) if self.conv_mean_hist else 0.0
            avg_conv_last_3 = float(np.mean(self.conv_mean_hist[-3:])) if self.conv_mean_hist else 0.0
            avg_xi_all = float(np.mean(xi_mean[:t])) if t > 0 else 0.0
            avg_xi_last_3 = _mean_slice(xi_mean, t, 3)

            pvalue_agg = float(pvalue_mean[t]) if t < len(pvalue_mean) else 0.0
            timeStepIndex_volume_agg = float(volume[t]) if t < len(volume) else 0.0
            last_3_timeStepIndexs_volume = float(last3_volume[t]) if t < len(last3_volume) else 0.0
            historical_volume = float(hist_volume[t]) if t < len(hist_volume) else 0.0

            current_state = np.array([
                timeleft, bgtleft,
                avg_bid_all, avg_bid_last_3,
                avg_lwc_all, avg_pvalue_all, avg_conv_all, avg_xi_all,
                avg_lwc_last_3, avg_pvalue_last_3, avg_conv_last_3, avg_xi_last_3,
                pvalue_agg, timeStepIndex_volume_agg, last_3_timeStepIndexs_volume, historical_volume
            ], dtype=np.float32)
        else:
            # Fallback: simplified 8-d features.
            budget_usage = 1.0 - (self.remaining_budget / self.budget)

            if len(historyBids) > 0:
                avg_bid = np.mean(historyBids[-1])
                win_rate = np.mean([r[0] for r in historyAuctionResult[-1]])
                cvr = np.mean([r[0] for r in historyImpressionResult[-1]])
            else:
                avg_bid = 0.0
                win_rate = 0.0
                cvr = 0.0

            current_state = np.zeros(self.config['state_dim'])
            current_state[0] = budget_usage
            current_state[1] = self.remaining_budget
            current_state[2] = self.budget
            current_state[3] = timeStep_index
            current_state[4] = self.cpa
            current_state[5] = avg_bid
            current_state[6] = win_rate
            current_state[7] = cvr

        # Normalize.
        state_norm = (current_state - self.state_mean) / (self.state_std + 1e-8)
        state_tensor = torch.from_numpy(state_norm).float().to(self.device)

        # Generate language embeddings.
        if len(historyBids) > 0:
            suggested_bid = float(np.mean(historyBids[-1]))
        else:
            suggested_bid = 10.0
        lang_task_desc_emb, lang_history_emb, lang_strategy_emb = self._get_language_embeddings(current_state, suggested_bid)

        # Update history buffers.
        self.states.append(state_tensor)
        self.timesteps.append(timeStep_index)
        self.lang_task_desc.append(lang_task_desc_emb)
        self.lang_history.append(lang_history_emb)
        self.lang_strategy.append(lang_strategy_emb)

        # Keep max length.
        if len(self.states) > self.max_length:
            self.states = self.states[-self.max_length:]
            self.timesteps = self.timesteps[-self.max_length:]
            self.lang_task_desc = self.lang_task_desc[-self.max_length:]
            self.lang_history = self.lang_history[-self.max_length:]
            self.lang_strategy = self.lang_strategy[-self.max_length:]
            if len(self.actions) > self.max_length - 1:
                self.actions = self.actions[-(self.max_length-1):]

        # Prepare model inputs.
        seq_len = len(self.states)
        states_seq = torch.stack(self.states).unsqueeze(0)
        timesteps_seq = torch.tensor([self.timesteps], dtype=torch.long, device=self.device)
        rtg_seq = torch.ones((1, seq_len, 1), device=self.device) * self.rtg_value

        if len(self.actions) > 0:
            actions_seq = torch.stack(self.actions).unsqueeze(0)
            if actions_seq.shape[1] < seq_len:
                pad_len = seq_len - actions_seq.shape[1]
                actions_seq = torch.cat([
                    actions_seq,
                    torch.zeros((1, pad_len, self.config['act_dim']), device=self.device)
                ], dim=1)
        else:
            actions_seq = torch.zeros((1, seq_len, self.config['act_dim']), device=self.device)

        rewards_seq = torch.zeros((1, seq_len, 1), device=self.device)
        lang_task_desc_seq = torch.stack(self.lang_task_desc).unsqueeze(0)
        lang_history_seq = torch.stack(self.lang_history).unsqueeze(0)
        lang_strategy_seq = torch.stack(self.lang_strategy).unsqueeze(0)

        # Model inference.
        with torch.no_grad():
            action_preds = self.model.forward(
                states=states_seq,
                actions=actions_seq,
                rewards=rewards_seq,
                returns_to_go=rtg_seq,
                timesteps=timesteps_seq,
                language_task=lang_task_desc_seq,
                language_history=lang_history_seq,
                language_strategy=lang_strategy_seq
            )
            action = action_preds[0, -1, 0].item()

        # Post-process.
        action = action * self.action_scale
        action = np.clip(action, 0, self.action_clip)
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.float32)
        self.actions.append(action_tensor)

        self.last_state = current_state
        self.last_action = action

        # Compute bid.
        bid = action * pValue
        return bid


def getScore_nips(reward, cpa, cpa_constraint):
    """NIPS score function."""
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def load_language_generator(template_module):
    module = importlib.import_module(template_module)
    if not hasattr(module, "BiddingLanguageGeneratorWithTask"):
        raise ValueError(f"Template module missing BiddingLanguageGeneratorWithTask: {template_module}")
    return module.BiddingLanguageGeneratorWithTask


def run_test(model_dir, test_file, budget_ratio=1.0, language_emb_dim=2048,
             embedding_lookup=None, language_generator_cls=None):
    """Run standard test with GAS/GAVE-style budget handling."""
    logger.info("="*80)
    logger.info("Qwen-0.5B + 2048-d Test (standard, History/Strategy before State)")
    logger.info(f"Model: {os.path.basename(model_dir)}")
    logger.info(f"Test data: {os.path.basename(test_file)}")
    logger.info("="*80)

    # Load test data.
    data_loader = TestDataLoader(file_path=test_file)
    env = OfflineEnv()

    keys = data_loader.keys
    total_score = 0.0
    total_reward = 0.0
    total_cost = 0.0
    exceed_count = 0

    # Shared Qwen encoder (only needed without lookup table).
    shared_encoder = None
    if embedding_lookup is None:
        logger.info("Initializing shared Qwen-0.5B encoder (2048-d)...")
        shared_encoder = QwenEncoder2048(output_dim=language_emb_dim)
        logger.info("Shared encoder ready")
    else:
        logger.info("Using precomputed embedding lookup")

    for idx, key in enumerate(keys):
        logger.info(f"\n[{idx+1}/{len(keys)}] Testing agent: {key}")

        # Fetch test data.
        num_steps, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)
        adjusted_budget = budget * budget_ratio

        # Precompute stats aligned with training (from test data).
        group = data_loader.test_dict[key].sort_values('timeStepIndex')
        agg_cols = {'pValue': 'mean', 'leastWinningCost': 'mean'}
        if 'xi' in group.columns:
            agg_cols['xi'] = 'mean'
        agg = group.groupby('timeStepIndex').agg(agg_cols)
        volume = group.groupby('timeStepIndex').size()

        pvalue_mean = np.zeros(num_steps, dtype=np.float32)
        lwc_mean = np.zeros(num_steps, dtype=np.float32)
        xi_mean = np.zeros(num_steps, dtype=np.float32)
        vol_arr = np.zeros(num_steps, dtype=np.float32)

        for t, row in agg.iterrows():
            tt = int(t)
            if tt >= num_steps:
                continue
            pvalue_mean[tt] = float(row['pValue'])
            lwc_mean[tt] = float(row['leastWinningCost'])
            if 'xi' in row:
                xi_mean[tt] = float(row['xi'])
        for t, v in volume.items():
            tt = int(t)
            if tt < num_steps:
                vol_arr[tt] = float(v)

        historical_volume = np.zeros(num_steps, dtype=np.float32)
        last3_volume = np.zeros(num_steps, dtype=np.float32)
        cum = 0.0
        for t in range(num_steps):
            historical_volume[t] = cum
            last3_volume[t] = float(np.sum(vol_arr[max(0, t-3):t]))
            cum += vol_arr[t]

        # Initialize policy.
        agent = BiddingStrategy(
            model_dir=model_dir,
            budget=adjusted_budget,
            cpa=cpa,
            category=category,
            shared_encoder=shared_encoder,
            language_emb_dim=language_emb_dim,
            embedding_lookup=embedding_lookup,
            language_generator_cls=language_generator_cls
        )
        if getattr(agent, "rtg_mode", "fixed") == "budget_cpa":
            agent.rtg_value = adjusted_budget / (cpa + 1e-10)
        agent.precomputed_stats = {
            "pvalue_mean": pvalue_mean,
            "lwc_mean": lwc_mean,
            "xi_mean": xi_mean,
            "volume": vol_arr,
            "historical_volume": historical_volume,
            "last3_volume": last3_volume
        }

        # History buffers.
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': []
        }

        rewards = np.zeros(num_steps)

        for t in range(num_steps):
            pValue = pValues[t]
            pValueSigma = pValueSigmas[t]
            leastWinningCost = leastWinningCosts[t]

            # Agent bidding decision.
            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid = agent.bidding(
                    t, pValue, pValueSigma,
                    history['historyBids'],
                    history['historyAuctionResult'],
                    history['historyImpressionResult'],
                    [leastWinningCost]
                )

            # Environment simulation.
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                pValue, pValueSigma, bid, leastWinningCost)

            # Budget handling (GAS/GAVE-style).
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                if len(pv_index) == 0:
                    break
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                    pValue, pValueSigma, bid, leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            # Update budget and stats.
            agent.remaining_budget -= np.sum(tick_cost)
            rewards[t] = np.sum(tick_conversion)
            agent.last_reward = rewards[t]
            # Update history stats used for state construction.
            agent.bid_mean_hist.append(float(np.mean(bid)) if len(bid) > 0 else 0.0)
            agent.conv_mean_hist.append(float(np.mean(tick_conversion)) if len(tick_conversion) > 0 else 0.0)

            # Record history.
            history['historyBids'].append(bid)
            history['historyAuctionResult'].append(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(len(tick_status))])
            history['historyImpressionResult'].append(
                [(tick_conversion[i], tick_conversion[i]) for i in range(len(tick_conversion))])

        # Compute metrics.
        all_reward = np.sum(rewards)
        all_cost = adjusted_budget - agent.remaining_budget
        cpa_real = np.clip(all_cost / (all_reward + 1e-10), 0, 100)
        score = getScore_nips(all_reward, cpa_real, cpa)

        # Aggregate stats.
        total_score += score
        total_reward += all_reward
        total_cost += all_cost
        if cpa_real > cpa:
            exceed_count += 1

        logger.info(f"  Budget: {adjusted_budget:.2f} | CPA limit: {cpa:.2f}")
        logger.info(f"  Conversions: {all_reward:.2f} | Cost: {all_cost:.2f}")
        logger.info(f"  CPA actual: {cpa_real:.2f} | Score: {score:.2f}")

    # Final results.
    num_agents = len(keys)
    avg_score = total_score / num_agents
    avg_cpa = total_cost / (total_reward + 1e-10)
    exceed_rate = exceed_count / num_agents * 100

    logger.info("\n" + "="*80)
    logger.info("Final test results")
    logger.info("="*80)
    logger.info(f"Average score: {avg_score:.2f}")
    logger.info(f"Total conversions: {total_reward:.2f}")
    logger.info(f"Total cost: {total_cost:.2f}")
    logger.info(f"Average CPA: {avg_cpa:.2f}")
    logger.info(f"CPA exceed rate: {exceed_rate:.1f}% ({exceed_count}/{num_agents})")
    logger.info("="*80)

    return {
        'avg_score': avg_score,
        'total_reward': total_reward,
        'total_cost': total_cost,
        'avg_cpa': avg_cpa,
        'exceed_rate': exceed_rate
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qwen-0.5B + 2048-d test (standard)')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory path')
    parser.add_argument('--budget_ratio', type=float, default=1.0, help='Budget ratio')
    parser.add_argument('--result_suffix', type=str, default="",
                        help='Result file suffix (for multiple budget runs)')
    parser.add_argument('--test_file', type=str,
                       default='data/test/period-7.csv',
                       help='Test data file')
    parser.add_argument('--language_emb_dim', type=int, default=2048, help='Language embedding dimension')
    parser.add_argument('--embedding_lookup', type=str, default=None, help='Embedding lookup path')
    parser.add_argument('--template_module', type=str, default='language_templates_high',
                        help='Template module (language_templates_high or language_templates_low)')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Qwen-0.5B + 2048-d test start (standard)")
    logger.info("="*80)

    # Load embedding lookup if provided.
    embedding_lookup = None
    if args.embedding_lookup and os.path.exists(args.embedding_lookup):
        logger.info(f"Loading embedding lookup: {args.embedding_lookup}")
        with open(args.embedding_lookup, 'rb') as f:
            embedding_lookup = pickle.load(f)
        logger.info("Embedding lookup loaded")

    # Load template generator.
    language_generator_cls = load_language_generator(args.template_module)

    start_time = time.time()
    results = run_test(
        args.model_dir,
        args.test_file,
        args.budget_ratio,
        args.language_emb_dim,
        embedding_lookup,
        language_generator_cls,
    )
    elapsed = time.time() - start_time

    logger.info(f"\nTotal test time: {elapsed:.2f}s")
    logger.info("Test finished")

    # Save results.
    suffix = args.result_suffix.strip()
    if suffix:
        result_file = os.path.join(args.model_dir, f'test_results_standard_{suffix}.json')
    else:
        result_file = os.path.join(args.model_dir, 'test_results_standard.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {result_file}")
