
from __future__ import annotations
import os
import time
import yaml
import torch
import numpy as np
from src.brokers.alpaca import AlpacaPaper
from src.models.q_network import DuelingQNet, QNet

A_HOLD, A_LONG, A_FLAT = 0, 1, 2


def load_cfg(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def decide_action(qnet, state):
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0)  # [1, C, T, F]
        q = qnet(s)
        return int(q.argmax(dim=1).item())


def main():
    cfg = load_cfg()
    sym = cfg["live"]["symbol"]
    qty = int(cfg["live"]["qty"])

    # Load a trained model
    obs_shape = (1, cfg["features"]["window"], 10)
    Net = DuelingQNet if cfg["agent"]["dueling"] else QNet
    qnet = Net(obs_shape, 3, tuple(cfg["agent"]["hidden_dims"]))
    qnet.load_state_dict(torch.load("data/runs/latest/checkpoints/final_policy.pt", map_location="cpu"))
    qnet.eval()

    broker = AlpacaPaper()

    from collections import deque
    import pandas as pd

    # rolling window state buffer (mock using recent quotes only; in practice, recompute full feature set)
    winT = cfg["features"]["window"]
    F = 10
    buf = deque(maxlen=winT)

    while True:
        price = float(broker.get_quote(sym))
        row = np.array([price, 0, 0, 50, 0, 0, 0, 50, 50, 0.5], dtype=np.float32)  # placeholder live features
        buf.append(row)
        if len(buf) == winT:
            arr = np.stack(buf)
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-8)
            state = np.expand_dims(arr, 0)  # [C=1, T, F]
            a = decide_action(qnet, state)
            if a == A_LONG:
                print("BUY", broker.submit_order(sym, qty, "buy"))
            elif a == A_FLAT:
                print("SELL", broker.submit_order(sym, qty, "sell"))
            else:
                print("HOLD")
        time.sleep(cfg["live"]["poll_seconds"])

if __name__ == "__main__":
    main()