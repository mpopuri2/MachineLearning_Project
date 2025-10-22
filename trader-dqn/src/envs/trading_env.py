from __future__ import annotations
import numpy as np
import pandas as pd

A_HOLD, A_LONG, A_FLAT = 0, 1, 2  # discrete actions


class TradingEnv:
    """
    Simple long/flat trading environment:
      - Action 0: HOLD (keep position)
      - Action 1: LONG (enter or keep long)
      - Action 2: FLAT (exit to cash)

    Position is either 0 (flat) or 1 (long).  
    Reward is risk-adjusted PnL = ΔEquity / (rolling volatility + ε)
    """
    def __init__(self, prices: pd.Series, features: np.ndarray, initial_cash=100000,
                 transaction_cost_bp=5, stop_loss_pct=0.05, take_profit_pct=0.10):
        assert len(prices) == features.shape[0], "align features & prices"
        self.prices = prices.values.astype(np.float32)
        self.features = features.astype(np.float32)
        self.n_steps = len(self.prices)
        self.initial_cash = initial_cash
        self.tc = transaction_cost_bp / 1e4
        self.sl = stop_loss_pct
        self.tp = take_profit_pct
        self.reset()

    @property
    def obs_shape(self):
        # convert [T, F] window into [C=1, T, F]
        return (1, self.features.shape[1], self.features.shape[2])

    @property
    def n_actions(self):
        return 3

    def reset(self):
        self.t = 0
        self.pos = 0  # 0 = flat, 1 = long
        self.entry_price = None
        self.cash = float(self.initial_cash)
        self.equity = float(self.initial_cash)
        self.equity_window = []  # track short-term equity changes
        return self._obs()

    def step(self, action: int):
        assert 0 <= action < self.n_actions
        done = False
        price = self.prices[self.t]
        reward = 0.0

        # --- Check stop-loss / take-profit exits ---
        if self.pos == 1 and self.entry_price is not None:
            chg = (price - self.entry_price) / self.entry_price
            if chg <= -self.sl or chg >= self.tp:
                action = A_FLAT

        # --- Execute trade logic ---
        if action == A_LONG and self.pos == 0:
            cost = price * (1 + self.tc)
            if self.cash >= cost:
                self.cash -= cost
                self.pos = 1
                self.entry_price = price
        elif action == A_FLAT and self.pos == 1:
            proceeds = price * (1 - self.tc)
            self.cash += proceeds
            self.pos = 0
            self.entry_price = None
        # HOLD => nothing

        # --- Mark-to-market valuation ---
        mtm = self.pos * price
        prev_equity = self.equity
        self.equity = self.cash + mtm
        equity_change = self.equity - prev_equity

        # --- Risk-adjusted reward ---
        self.equity_window.append(equity_change)
        if len(self.equity_window) > 20:  # 20-step rolling window
            self.equity_window.pop(0)

        window_len = max(10, min(50, int(self.t * 0.05)))
        rolling_std = np.std(self.equity_window[-window_len:]) if len(self.equity_window) > 1 else 1.0
        reward = equity_change / (rolling_std + 1e-8)

        # --- Advance time ---
        self.t += 1
        if self.t >= self.n_steps - 1:
            done = True
            if self.pos == 1:
                price = self.prices[self.t]
                proceeds = price * (1 - self.tc)
                self.cash += proceeds
                self.pos = 0
                self.entry_price = None
                self.equity = self.cash
                equity_change = proceeds - price
                self.equity_window.append(equity_change)
                reward += equity_change / (rolling_std + 1e-8)

        return self._obs(), reward, done, {
            "t": self.t,
            "price": price,
            "pos": self.pos,
            "cash": self.cash,
            "equity": self.equity
        }

    def _obs(self):
        # Observation at time t
        obs = self.features[self.t]
        return np.expand_dims(obs, axis=0)  # [1, T, F]