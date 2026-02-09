"""
Token order: [Task, RTG, History, Strategy, State, Action]
"""
import torch
import torch.nn as nn
import os
import sys

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from dt import Block

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None

class LanguageGuidedDTWithTaskFlexible(nn.Module):
    def __init__(self, state_dim, act_dim, state_mean, state_std, 
                 use_language=True, language_emb_dim=768,  # configurable embedding dim
                 language_model_name="paraphrase-TinyBERT-L6-v2",
                 action_tanh=False, K=20, max_ep_len=96, scale=3000, target_return=8,
                 use_precomputed_embeddings=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_language = use_language
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.language_emb_dim = language_emb_dim  # store embedding dim
        self.hidden_size = 128
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device) if not isinstance(state_mean, torch.Tensor) else state_mean.to(self.device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device) if not isinstance(state_std, torch.Tensor) else state_std.to(self.device)
        
        self.max_length = K
        self.max_ep_len = max_ep_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.length_times = 6 if use_language else 3

        # Optimization setup
        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 0.0001

        # Transformer block config
        block_config = {
            "n_ctx": 1024,
            "n_embd": 128,
            "n_layer": 6,
            "n_head": 4,
            "n_inner": 1024,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "block_size": 1024
        }
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])
        
        # Embeddings
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_reward = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_size)
        
        # Language components (support flexible embedding size)
        if self.use_language:
            # Load the language encoder only when embeddings are not precomputed.
            if not use_precomputed_embeddings:
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers is required for language mode")
                
                print(f"Loading language encoder: {language_model_name}...")
                self.language_encoder = SentenceTransformer(f"sentence-transformers/{language_model_name}")
                self.language_encoder.eval()
                for param in self.language_encoder.parameters():
                    param.requires_grad = False
            else:
                print(f"Using precomputed embeddings ({language_emb_dim}D); skipping encoder loading.")
                self.language_encoder = None
            
            # Project language embeddings to model hidden size.
            # Task + History + Strategy
            print(f"Initializing language projection: {language_emb_dim} -> {self.hidden_size}")
            self.embed_language_task = nn.Linear(language_emb_dim, self.hidden_size)
            self.embed_language_h = nn.Linear(language_emb_dim, self.hidden_size)
            self.embed_language_f = nn.Linear(language_emb_dim, self.hidden_size)
            
            # Empty language embedding for padding.
            if not use_precomputed_embeddings and self.language_encoder is not None:
                with torch.no_grad():
                    empty_emb = self.language_encoder.encode("", convert_to_tensor=True, device=self.device)
                    if isinstance(empty_emb, torch.Tensor):
                        self.empty_language_emb = empty_emb.to(self.device)
                    else:
                        self.empty_language_emb = torch.tensor(empty_emb, dtype=torch.float32, device=self.device)
            else:
                # For precomputed embeddings, use a zero vector as padding.
                self.empty_language_emb = torch.zeros(language_emb_dim, dtype=torch.float32, device=self.device)
            
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(self.hidden_size, 1)
        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / self.warmup_steps, 1)
        )
        
        self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, timesteps, 
                language_task=None, language_history=None, language_strategy=None, attention_mask=None):
        """
        Forward pass.
        Token order: [Task, RTG, History, Strategy, State, Action].
        Predict action from the State token (index 4).
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # During training, RTG is shifted by one step, so lengths may differ.
        rtg_seq_length = returns_to_go.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, rtg_seq_length), dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask[:, :rtg_seq_length]
        
        # Truncate inputs to a common length.
        states = states[:, :rtg_seq_length, :]
        actions = actions[:, :rtg_seq_length, :]
        rewards = rewards[:, :rtg_seq_length, :]
        timesteps = timesteps[:, :rtg_seq_length]
        
        # Base embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Language embeddings (if enabled)
        if self.use_language:
            # Task Description embedding
            if language_task is not None and language_task.numel() > 0:
                if language_task.dim() == 2:
                    language_task = language_task.unsqueeze(1)  # [batch, 1, emb_dim]
                # Truncate to match RTG length (shifted during training).
                language_task = language_task[:, :rtg_seq_length, :]
                task_embeddings = self.embed_language_task(language_task)
                task_embeddings = task_embeddings + time_embeddings
            else:
                task_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # History embedding
            if language_history is not None and language_history.numel() > 0:
                if language_history.dim() == 2:
                    language_history = language_history.unsqueeze(1)  # [batch, 1, emb_dim]
                # Truncate to the correct length.
                language_history = language_history[:, :rtg_seq_length, :]
                history_embeddings = self.embed_language_h(language_history)
                history_embeddings = history_embeddings + time_embeddings
            else:
                history_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # Strategy embedding
            if language_strategy is not None and language_strategy.numel() > 0:
                if language_strategy.dim() == 2:
                    language_strategy = language_strategy.unsqueeze(1)  # [batch, 1, emb_dim]
                # Truncate to the correct length.
                language_strategy = language_strategy[:, :rtg_seq_length, :]
                strategy_embeddings = self.embed_language_f(language_strategy)
                strategy_embeddings = strategy_embeddings + time_embeddings
            else:
                strategy_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # Token order: [Task, RTG, History, Strategy, State, Action]
            stacked_inputs = torch.stack(
                (task_embeddings, returns_embeddings, history_embeddings,
                 strategy_embeddings, state_embeddings, action_embeddings),
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 6*rtg_seq_length, self.hidden_size)
            
            # Expand attention mask (6 tokens per step).
            stacked_attention_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask, 
                 attention_mask, attention_mask, attention_mask], 
                dim=1
            ).permute(0, 2, 1).reshape(batch_size, 6*rtg_seq_length)
        else:
            # No language: [RTG, State, Action]
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), 
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*rtg_seq_length, self.hidden_size)
            
            stacked_attention_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask], 
                dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*rtg_seq_length)
        
        # Transformer forward
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Use the stacked attention mask (same as original implementation).
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, rtg_seq_length), device=self.device)
        
        attention_mask = attention_mask.to(self.device)
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * rtg_seq_length)
        stacked_attention_mask = stacked_attention_mask.to(device=stacked_inputs.device, dtype=stacked_inputs.dtype)
        
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)
        
        # Predict action from the State token (index 4).
        if self.use_language:
            x = x.reshape(batch_size, rtg_seq_length, 6, self.hidden_size).permute(0, 2, 1, 3)
            state_preds = x[:, 4]  # State token
        else:
            x = x.reshape(batch_size, rtg_seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
            state_preds = x[:, 1]  # State token in non-language mode
        
        # Action prediction
        action_preds = self.predict_action(state_preds)
        
        return action_preds

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask,
             language_task=None, language_history=None, language_strategy=None):
        """Training step: compute loss and update parameters."""
        # Predict current action using shifted inputs (DT-style).
        action_target = actions[:, :-1]
        action_preds = self.forward(
            states, actions, rewards, rtg[:, :-1], timesteps,
            language_task=language_task,
            language_history=language_history,
            language_strategy=language_strategy,
            attention_mask=attention_mask
        )

        # Compute loss
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask[:, :-1].reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask[:, :-1].reshape(-1) > 0]
        
        loss = torch.mean((action_preds - action_target) ** 2)
        
        # Backprop and optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.detach().cpu().item()

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   language_task=None, language_history=None, language_strategy=None):
        """Inference: get next action."""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        rewards = rewards.reshape(1, -1, 1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]

        action_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps,
            language_task=language_task,
            language_history=language_history,
            language_strategy=language_strategy
        )
        
        return action_preds[0, -1].cpu().numpy()

    def init_eval(self):
        """Initialize evaluation caches."""
        self.eval_states = None
        self.eval_actions = None
        self.eval_rewards = None
        self.eval_returns_to_go = None
        self.eval_timesteps = None
        self.eval_language_task = None
        self.eval_language_h = None
        self.eval_language_f = None
