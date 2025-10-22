import numpy as np
from src.envs.trading_env import TradingEnv

def test_env_shapes():
    prices = np.linspace(100, 110, 200)
    feats = np.random.randn(200, 50, 10).astype('float32')
    env = TradingEnv(prices=prices, features=feats)
    s = env.reset()
    assert s.shape == (1, 50, 10)
    ns, r, d, info = env.step(1)
    assert isinstance(r, float)
    assert 'equity' in info