import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class LogStats(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        if not done:
            return True
        for k, v in self.locals['env'].env.flush_stats().items():
            self.logger.record_mean(f'rollout/{k}', v)
        return True
