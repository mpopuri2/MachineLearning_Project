from __future__ import annotations
import os
import yaml
import torch
import numpy as np
import pandas as pd
from src.logger import Logger
from src.data.yfinance_loader import download_ohlcv
from src.feature_pipeline import build_features, make_state_matrix
from src.envs.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent

torch.backends.mps.enabled = True
torch.backends.mps.allow_tf32 = True
torch.set_float32_matmul_precision("high")
def load_cfg(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    logger = Logger()

    sym = cfg["symbols"][0]
    print(f"[INFO] Downloading data for {sym} ...")
    raw = download_ohlcv(sym, cfg["train_period"]["start"], cfg["train_period"]["end"], cfg["interval"])

    print("[INFO] Building features ...")
    feat = build_features(raw)

    window = cfg["features"]["window"]
    if len(feat) < window:
        raise RuntimeError(
            f"Insufficient data after feature building: len(df)={len(feat)} < window={window}. "
            "Check build_features for excessive NaN removal."
        )

    cols = [
    "close","volume","ret1","rsi","macd","macd_signal","macd_hist",
    "stoch_k","stoch_d","bb_pct","obv","ema_20","ema_50","atr"
    ]

    states = make_state_matrix(feat, cols, window, cfg["features"]["normalize"])

    # Align prices to state length
    N_states = states.shape[0]
    prices = feat["close"].iloc[-N_states:].reset_index(drop=True)
    if len(prices) != N_states:
        raise RuntimeError(f"Data mismatch: prices len ({len(prices)}) != states ({N_states})")

    print(f"[INFO] Created {N_states} states with {states.shape[2]} features per step.")

    env = TradingEnv(
        prices,
        states,
        initial_cash=cfg["env"]["initial_cash"],
        transaction_cost_bp=cfg["env"]["transaction_cost_bp"],
        stop_loss_pct=cfg["env"]["stop_loss_pct"],
        take_profit_pct=cfg["env"]["take_profit_pct"],
    )

    agent = DQNAgent(env.obs_shape, env.n_actions, cfg["agent"])
    state = env.reset()

    losses, equities = [], []
    eval_steps = cfg["training"]["eval_every_steps"]
    checkpoint_steps = cfg["training"]["checkpoint_every_steps"]

    print("[INFO] Starting DQN training loop ...")
    for step in range(cfg["training"]["max_steps"]):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.push(state, action, reward, next_state, float(done))
        agent.total_steps += 1

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        agent.maybe_update_target()
        state = next_state
        equities.append(info.get("equity", env.equity))

        if done:
            state = env.reset()

        # ---- Evaluation ----
        if (step + 1) % eval_steps == 0:
            avg_loss = np.mean(losses[-eval_steps:]) if losses else 0
            eq = equities[-1]
            if isinstance(eq, np.ndarray):
                eq = eq.item()
            else:
                eq = float(eq)
            print(
                f"Step {step + 1:06d} | "
                f"Equity={eq:10.2f} | "
                f"Avg Loss={avg_loss:.5f} | "
                f"Epsilon={agent.epsilon():.4f}"
            )

        # ---- Checkpoint ----
        if (step + 1) % checkpoint_steps == 0:
            p = logger.path("checkpoints", f"policy_step_{step + 1}.pt")
            torch.save(agent.policy.state_dict(), p)

    # ---- Final Save ----
    final_eq = equities[-1]
    if isinstance(final_eq, np.ndarray):
        final_eq = float(final_eq)
    torch.save(agent.policy.state_dict(), logger.path("checkpoints", "final_policy.pt"))
    print(f"\n Training complete. Final equity: {final_eq:.2f}")


if __name__ == "__main__":
    main()