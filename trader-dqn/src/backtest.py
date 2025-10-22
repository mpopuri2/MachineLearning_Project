from __future__ import annotations
import numpy as np


def backtest(prices, actions, initial_cash=100000, tc=0.0001, slippage=0.0001):
    cash = initial_cash
    pos = 0
    equity_curve = []
    for t in range(len(prices)):
        price = prices[t]
        # execute actions (A_LONG=1, A_FLAT=2)
        if actions[t] == 1 and pos == 0:  # buy
            cash -= price * (1 + tc + slippage)
            pos = 1
        elif actions[t] == 2 and pos == 1:  # sell
            cash += price * (1 - tc - slippage)
            pos = 0
        equity_curve.append(cash + pos * price)
    return np.array(equity_curve)