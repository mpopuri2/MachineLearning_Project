from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.replay_buffer import ReplayBuffer
from src.models.q_network import QNet, DuelingQNet


class DQNAgent:
    def __init__(self, obs_shape, n_actions, cfg):

        # -------- Force CPU only ----------
        self.device = torch.device("cpu")
        print(f"[INFO] Using device: CPU only")

        # -------- Network setup ----------
        self.n_actions = int(n_actions)
        Net = DuelingQNet if cfg.get("dueling", True) else QNet
        hidden_dims = tuple(cfg.get("hidden_dims", [256, 256]))

        self.policy = Net(obs_shape, self.n_actions, hidden_dims).to(self.device)
        self.target = Net(obs_shape, self.n_actions, hidden_dims).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        # -------- Optimizer & hyperparameters ----------
        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(cfg.get("lr", 5e-4)))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.buffer = ReplayBuffer(int(cfg.get("buffer_size", 200_000)))
        self.learn_start = int(cfg.get("learn_start", 5_000))
        self.target_update_every = int(cfg.get("target_update_every", 2_000))
        self.double_dqn = bool(cfg.get("double_dqn", True))

        # Epsilon schedule for exploration
        self.eps_start = float(cfg.get("epsilon_start", 1.0))
        self.eps_end = float(cfg.get("epsilon_end", 0.05))
        self.eps_decay_steps = int(cfg.get("epsilon_decay_steps", 200_000))
        self.total_steps = 0

    # ---------- Epsilon schedule ----------
    def epsilon(self) -> float:
        """Linear decay of epsilon over time."""
        frac = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ---------- Action selection ----------
    def act(self, state: np.ndarray) -> int:
        """
        Selects an action given a state using epsilon-greedy policy.
        state: np.ndarray shape [1, C, T, F]
        """
        if np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            s = torch.from_numpy(state).to(self.device, dtype=torch.float32)
            q = self.policy(s)
            return int(q.argmax(dim=1).item())

    # ---------- Replay Buffer ----------
    def push(self, *args):
        """Push transition to buffer."""
        self.buffer.push(*args)

    # ---------- Learning / Backprop ----------
    def learn(self):
        """Performs one gradient step if enough samples exist."""
        if len(self.buffer) < max(self.learn_start, self.batch_size):
            return None

        # Sample mini-batch
        batch = self.buffer.sample(self.batch_size)

        # Convert numpy to torch tensors
        state = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(batch.action), dtype=torch.long, device=self.device).view(-1, 1)
        reward = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.tensor(np.array(batch.done), dtype=torch.float32, device=self.device).view(-1, 1)

        # Q(s, a)
        q = self.policy(state).gather(1, action)

        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy(next_state).argmax(dim=1, keepdim=True)
                q_next = self.target(next_state).gather(1, next_actions)
            else:
                q_next = self.target(next_state).max(dim=1, keepdim=True).values
            target = reward + (1.0 - done) * self.gamma * q_next

        # Loss and optimization
        loss = nn.SmoothL1Loss()(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    # ---------- Target Network Update ----------
    def maybe_update_target(self, tau: float = 0.005):
        """Soft target network update."""
        with torch.no_grad():
            for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)