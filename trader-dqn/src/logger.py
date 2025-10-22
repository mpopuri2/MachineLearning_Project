import os
from datetime import datetime

class Logger:
    def __init__(self, base_dir="data/runs"):
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, ts)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.fig_dir = os.path.join(self.run_dir, "figs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def path(self, *parts):
        p = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p