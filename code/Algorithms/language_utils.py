"""
Language utilities for trajectory preprocessing and batching with Task support.
"""
import numpy as np
import pandas as pd
import torch
import pickle
import sys
from torch.utils.data import Dataset
from language_templates_high import BiddingLanguageGeneratorWithTask

class NumpyCompatUnpickler(pickle.Unpickler):
    """Map numpy._core.* pickles to numpy.core.* for numpy<2.0."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle_compat(path):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


class LanguageAugmentedReplayBufferWithTask(Dataset):
    """
    Language-augmented replay buffer with Task Description support.
    """
    def __init__(self, state_dim, act_dim, data_path, K=20, scale=3000, 
                 use_language=True, language_type='both', use_precomputed_embeddings=False):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.K = K
        self.scale = scale
        self.use_language = use_language
        self.language_type = language_type
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        # Initialize language generator.
        if self.use_language and not use_precomputed_embeddings:
            self.language_generator = BiddingLanguageGeneratorWithTask()
        
        # Load data.
        print(f"Loading data from {data_path}...")
        if data_path.endswith('.pkl') and use_precomputed_embeddings:
            # Load precomputed embeddings (new format).
            precomputed_data = load_pickle_compat(data_path)
            
            # Check for DataFrame format (new preprocessing output).
            if hasattr(precomputed_data, 'columns') and 'task_description_embeddings' in precomputed_data.columns:
                print("Loading precomputed embeddings from DataFrame format...")
                # Build trajectories directly from DataFrame.
                self.trajectories = self._load_from_csv_df(precomputed_data, use_precomputed=True)
                print(f"Loaded {len(self.trajectories)} trajectories with embeddings")
            elif isinstance(precomputed_data, dict) and 'task_embeddings' in precomputed_data:
                # Legacy dict format.
                print("Loading precomputed embeddings format...")
                csv_path = data_path.replace('_with_task.pkl', '.csv')
                print(f"Loading raw trajectories from {csv_path}...")
                self.trajectories = self._load_from_csv(csv_path, use_precomputed=False)
                
                # Assign precomputed embeddings to each trajectory.
                print("Assigning embeddings to trajectories...")
                task_embs = precomputed_data['task_embeddings']
                history_embs = precomputed_data['history_embeddings']
                strategy_embs = precomputed_data['strategy_embeddings']
                
                row_idx = 0
                for traj in self.trajectories:
                    traj_len = len(traj['states'])
                    traj['task_embeddings'] = task_embs[row_idx:row_idx+traj_len]
                    traj['history_embeddings'] = history_embs[row_idx:row_idx+traj_len]
                    traj['strategy_embeddings'] = strategy_embs[row_idx:row_idx+traj_len]
                    row_idx += traj_len
                
                print(f"Assigned embeddings to {len(self.trajectories)} trajectories")
            else:
                # Legacy format: list of trajectories.
                self.trajectories = precomputed_data
        elif data_path.endswith('.pkl'):
            self.trajectories = load_pickle_compat(data_path)
        else:
            self.trajectories = self._load_from_csv(data_path, use_precomputed=False)
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        
        # Compute normalization stats.
        all_states = np.concatenate([traj['states'] for traj in self.trajectories], axis=0)
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-6
        
        # Sampling weights.
        self.p_sample = self._compute_sampling_weights()
    
    def _load_from_csv_df(self, df, use_precomputed=False):
        """Load from DataFrame (supports precomputed embeddings)."""
        # Parse the state column if needed.
        if isinstance(df['state'].iloc[0], str):
            df['state'] = df['state'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        trajectories = []
        # Group by (advertiserNumber, deliveryPeriodIndex).
        grouped = df.groupby(['advertiserNumber', 'deliveryPeriodIndex'])

        for (adv_num, period_idx), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)

            states = np.array([row['state'] for _, row in group.iterrows()], dtype=np.float32)
            actions = group['action'].values.astype(np.float32)

            # Reward handling.
            if 'reward_continuous' in group.columns:
                rewards = group['reward_continuous'].values.astype(np.float32)
            elif 'reward' in group.columns:
                rewards = group['reward'].values.astype(np.float32)
            else:
                rewards = np.zeros(len(group), dtype=np.float32)

            # Compute RTG.
            rtgs = np.cumsum(rewards[::-1])[::-1]
            timesteps = np.clip(np.arange(len(states)), 0, 95)

            # Extract CPA for Task description.
            cpa_constraint = group['CPAConstraint'].iloc[0] if 'CPAConstraint' in group.columns else 10.0

            traj_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'rtgs': rtgs,
                'timesteps': timesteps,
                'cpa_constraint': cpa_constraint
            }

            # If precomputed embeddings exist, extract them.
            if use_precomputed:
                task_embs = np.array([row['task_description_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                history_embs = np.array([row['history_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                strategy_embs = np.array([row['strategy_embeddings'] for _, row in group.iterrows()], dtype=np.float32)

                traj_data['task_embeddings'] = task_embs
                traj_data['history_embeddings'] = history_embs
                traj_data['strategy_embeddings'] = strategy_embs

            trajectories.append(traj_data)

        return trajectories

    def _load_from_csv(self, csv_path, use_precomputed=False):
        """Load from CSV or PKL (supports preprocessed PKL)."""
        # Check if input is a PKL file.
        if csv_path.endswith('.pkl'):
            print(f"Loading preprocessed data from {csv_path}...")
            df = load_pickle_compat(csv_path)
            use_precomputed = 'task_description_embeddings' in df.columns
        else:
            df = pd.read_csv(csv_path)
            use_precomputed = False

            # Parse the state column.
            if 'state' in df.columns and isinstance(df['state'].iloc[0], str):
                df['state'] = df['state'].apply(lambda x: eval(x) if isinstance(x, str) else x)

        trajectories = []
        # Group by (advertiserNumber, deliveryPeriodIndex).
        grouped = df.groupby(['advertiserNumber', 'deliveryPeriodIndex'])

        for (adv_num, period_idx), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)
            
            states = np.array([row['state'] for _, row in group.iterrows()], dtype=np.float32)
            actions = group['action'].values.astype(np.float32)
            
            # Reward handling.
            if 'reward_continuous' in group.columns:
                rewards = group['reward_continuous'].values.astype(np.float32)
            elif 'reward' in group.columns:
                rewards = group['reward'].values.astype(np.float32)
            else:
                rewards = np.zeros(len(group), dtype=np.float32)
            
            # Compute RTG.
            rtgs = np.cumsum(rewards[::-1])[::-1]
            timesteps = np.clip(np.arange(len(states)), 0, 95)
            
            # Extract CPA for Task description.
            cpa_constraint = group['CPAConstraint'].iloc[0] if 'CPAConstraint' in group.columns else 10.0
            
            traj_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'rtgs': rtgs,
                'timesteps': timesteps,
                'cpa_constraint': cpa_constraint
            }
            
            # If precomputed embeddings exist, store them.
            if use_precomputed:
                traj_data['task_embeddings'] = np.array([row['task_description_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                traj_data['history_embeddings'] = np.array([row['history_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                traj_data['strategy_embeddings'] = np.array([row['strategy_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
            
            trajectories.append(traj_data)
        
        return trajectories
    
    def _compute_sampling_weights(self):
        """Compute sampling weights."""
        # Trajectory-length based sampling weights.
        traj_lens = np.array([len(traj['states']) for traj in self.trajectories])
        p_sample = traj_lens / np.sum(traj_lens)
        return p_sample
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """Get one sample (with Task)."""
        traj = self.trajectories[idx]
        traj_len = len(traj['states'])
        
        # Random start index.
        if traj_len > self.K:
            start_idx = np.random.randint(0, traj_len - self.K + 1)
            end_idx = start_idx + self.K
        else:
            start_idx = 0
            end_idx = traj_len
        
        # Extract sequence.
        states = traj['states'][start_idx:end_idx]
        actions = traj['actions'][start_idx:end_idx]
        rewards = traj['rewards'][start_idx:end_idx]
        rtgs = traj['rtgs'][start_idx:end_idx]
        timesteps = traj['timesteps'][start_idx:end_idx]
        cpa_constraint = traj['cpa_constraint']
        
        seq_len = end_idx - start_idx
        
        # Padding
        if seq_len < self.K:
            pad_len = self.K - seq_len
            states = np.concatenate([np.zeros((pad_len, self.state_dim)), states], axis=0)
            actions = np.concatenate([np.zeros(pad_len), actions], axis=0)
            rewards = np.concatenate([np.zeros(pad_len), rewards], axis=0)
            rtgs = np.concatenate([np.zeros(pad_len), rtgs], axis=0)
            timesteps = np.concatenate([np.zeros(pad_len), timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(pad_len), np.ones(seq_len)], axis=0)
        else:
            pad_len = 0  # No padding.
            attention_mask = np.ones(self.K)
        
        # Normalize states.
        states = (states - self.state_mean) / self.state_std
        
        # Normalize RTG.
        rtgs = rtgs / self.scale
        
        # Check for precomputed embeddings.
        has_precomputed = 'task_embeddings' in traj
        
        # Generate or use precomputed language embeddings.
        if self.use_language:
            if has_precomputed:
                # Use precomputed embeddings (fast, fixed).
                task_embs = traj['task_embeddings'][start_idx:end_idx]
                h_embs = traj['history_embeddings'][start_idx:end_idx]
                f_embs = traj['strategy_embeddings'][start_idx:end_idx]
                
                # Pad embeddings.
                if seq_len < self.K:
                    emb_dim = task_embs.shape[-1]
                    zero_emb = np.zeros((pad_len, emb_dim), dtype=np.float32)
                    task_embs = np.concatenate([zero_emb, task_embs], axis=0)
                    h_embs = np.concatenate([zero_emb, h_embs], axis=0)
                    f_embs = np.concatenate([zero_emb, f_embs], axis=0)
                
                return {
                    'states': torch.tensor(states, dtype=torch.float32),
                    'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                    'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                    'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                    'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                    'timesteps': torch.tensor(timesteps, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'lang_task_embs': torch.tensor(task_embs, dtype=torch.float32),
                    'lang_h_embs': torch.tensor(h_embs, dtype=torch.float32),
                    'lang_f_embs': torch.tensor(f_embs, dtype=torch.float32)
                }
            else:
                # Generate on the fly (keeps diversity).
                lang_task_texts = []
                lang_h_texts = []
                lang_f_texts = []
                
                for t in range(self.K):
                    if t < pad_len:
                        # Use empty strings for padding.
                        lang_task_texts.append("")
                        lang_h_texts.append("")
                        lang_f_texts.append("")
                    else:
                        actual_t = t - pad_len
                        # Task description (constant within a trajectory).
                        task_text = self.language_generator.generate_task_description(cpa_constraint)
                        
                        # History
                        if actual_t > 0:
                            prev_state = traj['states'][start_idx + actual_t - 1]
                            prev_action = traj['actions'][start_idx + actual_t - 1]
                            prev_reward = traj['rewards'][start_idx + actual_t - 1]
                            curr_state = traj['states'][start_idx + actual_t]
                            h_text = self.language_generator.generate_history(
                                prev_state, prev_action, prev_reward, curr_state
                            )
                        else:
                            h_text = ""
                        
                        # Strategy.
                        curr_state = traj['states'][start_idx + actual_t]
                        suggested_bid = traj['actions'][start_idx + actual_t - 1] if actual_t > 0 else 25.0
                        f_text = self.language_generator.generate_strategy(curr_state, suggested_bid)
                        
                        lang_task_texts.append(task_text)
                        lang_h_texts.append(h_text)
                        lang_f_texts.append(f_text)
                
                return {
                    'states': torch.tensor(states, dtype=torch.float32),
                    'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                    'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                    'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                    'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                    'timesteps': torch.tensor(timesteps, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'lang_task_texts': lang_task_texts,
                    'lang_h_texts': lang_h_texts,
                    'lang_f_texts': lang_f_texts
                }
        else:
            return {
                'states': torch.tensor(states, dtype=torch.float32),
                'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                'timesteps': torch.tensor(timesteps, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'lang_task_texts': None,
                'lang_h_texts': None,
                'lang_f_texts': None
            }


def language_collate_fn_with_task(batch):
    """
    Collate function (Task support, optional precomputed embeddings).
    """
    states = torch.stack([item['states'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    rewards = torch.stack([item['rewards'] for item in batch])
    dones = torch.stack([item['dones'] for item in batch])
    rtgs = torch.stack([item['rtgs'] for item in batch])
    timesteps = torch.stack([item['timesteps'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Check for precomputed embeddings.
    if 'lang_task_embs' in batch[0]:
        # Precomputed embeddings (return tensors).
        lang_task_embs = torch.stack([item['lang_task_embs'] for item in batch])
        lang_h_embs = torch.stack([item['lang_h_embs'] for item in batch])
        lang_f_embs = torch.stack([item['lang_f_embs'] for item in batch])
        return states, actions, rewards, dones, rtgs, timesteps, attention_mask, lang_task_embs, lang_h_embs, lang_f_embs
    else:
        # Text prompts (require online encoding).
        lang_task_texts = [item['lang_task_texts'] for item in batch] if batch[0]['lang_task_texts'] is not None else None
        lang_h_texts = [item['lang_h_texts'] for item in batch] if batch[0]['lang_h_texts'] is not None else None
        lang_f_texts = [item['lang_f_texts'] for item in batch] if batch[0]['lang_f_texts'] is not None else None
        return states, actions, rewards, dones, rtgs, timesteps, attention_mask, lang_task_texts, lang_h_texts, lang_f_texts
