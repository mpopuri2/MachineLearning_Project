import numpy as np
from src.agents.dqn_agent import DQNAgent

class Cfg(dict):
    pass

def test_agent_forward():
    obs_shape = (1, 50, 10)
    cfg = Cfg(
        gamma=0.99, lr=1e-3, batch_size=8, buffer_size=1000,
        learn_start=16, target_update_every=10,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100,
        hidden_dims=[64,64], dueling=True, double_dqn=True
    )
    agent = DQNAgent(obs_shape, 3, cfg)
    s = np.random.randn(1, *obs_shape).astype('float32')
    a = agent.act(s)
    assert a in {0,1,2}