from __future__ import annotations
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.data.yfinance_loader import download_ohlcv
from src.feature_pipeline import build_features, make_state_matrix
from src.envs.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.metrics import sharpe, max_drawdown


def load_cfg(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_latest_checkpoint() -> Path:
    runs = sorted(Path("data/runs").glob("*/checkpoints/final_policy.pt"))
    if not runs:
        raise FileNotFoundError("No trained model found in data/runs/. Please run `make train` first.")
    return runs[-1]


def find_close_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "close" in col.lower():
            return col
    raise KeyError("Could not find a column containing 'close' in features DataFrame.")


def evaluate():
    cfg = load_cfg()
    sym = cfg["symbols"][0]
    print(f"[INFO] Evaluating model on validation data for {sym}")

    # --- 1. Load Validation Data ---
    raw = download_ohlcv(sym, cfg["val_period"]["start"], cfg["val_period"]["end"], cfg["interval"])
    feat = build_features(raw)

    # --- Fix MultiIndex Columns from yfinance ---
    if isinstance(feat.columns, pd.MultiIndex):
        feat.columns = ['_'.join([c for c in col if c]) for col in feat.columns]
        feat.columns = [c.replace(f"{sym}_", "") for c in feat.columns]
        print(f"[INFO] Flattened MultiIndex columns (example: {feat.columns[:5].tolist()})")

    # --- Find correct close column dynamically ---
    close_col = find_close_column(feat)
    print(f"[INFO] Detected close column: '{close_col}'")

    # --- Select feature columns ---
    cols = [
        "close", "volume", "ret1", "rsi", "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "bb_high", "bb_low", "bb_pct",
        "obv", "ema_20", "ema_50", "atr"
    ]
    cols = [c for c in cols if c in feat.columns]
    if not cols:
        raise ValueError("No matching feature columns found in dataframe after preprocessing.")

    window = cfg["features"]["window"]
    states = make_state_matrix(feat, cols, window, cfg["features"]["normalize"])
    prices = feat[close_col].iloc[-states.shape[0]:].reset_index(drop=True)

    # --- 2. Create Validation Environment ---
    env = TradingEnv(
        prices,
        states,
        initial_cash=cfg["env"]["initial_cash"],
        transaction_cost_bp=cfg["env"]["transaction_cost_bp"],
        stop_loss_pct=cfg["env"]["stop_loss_pct"],
        take_profit_pct=cfg["env"]["take_profit_pct"]
    )

    # --- 3. Load Trained Model ---
    checkpoint_path = get_latest_checkpoint()
    agent = DQNAgent(env.obs_shape, env.n_actions, cfg["agent"])
    agent.policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    agent.policy.eval()
    print(f"[INFO] Loaded trained policy from: {checkpoint_path}")

    # --- 4. Evaluation Loop ---
    state = env.reset()
    equities, rets = [], []
    done = False
    prev_eq = env.equity

    while not done:
        with torch.no_grad():
            s = torch.from_numpy(state).float()
            q = agent.policy(s)
            a = int(q.argmax(dim=1).item())
        state, reward, done, info = env.step(a)
        eq = info["equity"]
        equities.append(float(eq))
        rets.append(float(eq - prev_eq))
        prev_eq = eq

    # --- 5. Compute Metrics ---
    daily_returns = np.array(rets)
    results = {
        "final_equity": float(equities[-1]),
        "roi_%": float((equities[-1] - equities[0]) / equities[0] * 100),
        "sharpe": float(np.nan_to_num(sharpe(daily_returns))),
        "max_drawdown": float(np.nan_to_num(max_drawdown(equities))),
    }

    print("\nEvaluation Metrics:")
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}")

    # --- 6. Plot Performance ---
    plt.figure(figsize=(10, 5))
    plt.plot(equities, label="Equity ($)", color="tab:blue")
    plt.title(f"Equity Curve - {sym}")
    plt.xlabel("Time Step")
    plt.ylabel("Equity ($)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()