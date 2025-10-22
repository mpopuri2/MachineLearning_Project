import numpy as np


def sharpe(returns, risk_free=0.0, eps=1e-8):
    r = np.array(returns)
    if r.size == 0:
        return 0.0
    excess = r - risk_free
    return float(excess.mean() / (excess.std() + eps) * np.sqrt(252))


def max_drawdown(equity_curve):
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    return float(dd.min())