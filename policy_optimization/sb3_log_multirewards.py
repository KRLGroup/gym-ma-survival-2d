import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class LogMultiRewards(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        done = self.locals['dones'][0]
        if not hasattr(self, 'rewards'):
            self.rewards = np.zeros_like(rewards)
        self.rewards += rewards
        if done:
            for i, r in enumerate(self.rewards):
                self.logger.record_mean(f'rollout/reward{i}', r)
            self.rewards = np.zeros_like(rewards)
        return True
