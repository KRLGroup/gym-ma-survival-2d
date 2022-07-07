from typing import Any, List, Union, Callable, Tuple, Optional, Dict

import numpy as np
import torch
import gym

from policy_optimization.utils import recursive_apply


def batch_observations(observations):
    return recursive_apply(lambda *xs: np.array(xs, dtype=xs[0].dtype), *observations)


# Tracks the state of N gym enviroments, and supports batched versions of Gym interface methods.
class EnvBatch:

    #TODO attr annotations

    #TODO support setting seeds for the first resets?

    def __init__(self, envs: List[gym.Env], dtype: torch.dtype = torch.float32, verbose: bool = False):
        self.envs = envs
        self.dtype = dtype
        self.verbose = verbose
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.observations = batch_observations([env.reset() for env in self.envs])
        # For consistency: this ensures that whenver done is 1, the 
        # corrsponding env was just reset.
        self.dones = torch.tensor(np.ones(len(self.envs)), dtype=dtype)
        self._steps = np.zeros(len(self.envs), dtype=int)

    def __len__(self):
        return len(self.envs)

    # Steps envs and resets the ones that finish. Returns the observations after the last, ignoring resets, to make it possible to retrieve observations right after terminal states.
    def step(self, actions: torch.Tensor, data: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actions = actions
        if data is not None:
            self.data = data
        steps = [env.step(actions[i].numpy()) for i, env in enumerate(self.envs)]
        observations = batch_observations([step[0] for step in steps])
        self.rewards = torch.tensor(np.array([step[1] for step in steps]), dtype=self.dtype)
        self.dones = torch.tensor(np.array([step[2] for step in steps]), dtype=self.dtype)
        self._steps += 1
        for i, done in enumerate(self.dones):
            if not done:
                continue
            if self.verbose:
                print(f'env {i} done after {self._steps[i]} steps')
            def f(x, y):
                x[i] = y
            recursive_apply(f, self.observations, self.envs[i].reset())
            self._steps[i] = 0
            #TODO do we also need to reset the done?
        return observations, self.rewards, self.dones

