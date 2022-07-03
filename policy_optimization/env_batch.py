from typing import Any, List, Union, Callable, Tuple, Optional, Dict

import numpy as np
import torch
import gym


# Tracks the state of N gym enviroments, and supports batched versions of Gym interface methods.
class EnvBatch:

    #TODO attr annotations

    #TODO support setting seeds for the first resets?

    def __init__(self, envs: List[gym.Env], dtype=torch.float32):
        self.envs = envs
        self.dtype = dtype
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.observations = torch.tensor(np.array([env.reset() for env in self.envs]), dtype=dtype)
        # For consistency: this ensures that whenver done is 1, the 
        # corrsponding env was just reset.
        self.dones = torch.tensor(np.ones(len(self.envs)), dtype=dtype)

    def __len__(self):
        return len(self.envs)

    # Steps envs and resets the ones that finish. Returns the observations after the last, ignoring resets, to make it possible to retrieve observations right after terminal states.
    def step(self, actions: torch.Tensor, data: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actions = actions
        if data is not None:
            self.data = data
        steps = [env.step(actions[i].numpy()) for i, env in enumerate(self.envs)]
        observations = torch.tensor(np.array([step[0] for step in steps]), dtype=self.dtype)
        self.rewards = torch.tensor([step[1] for step in steps], dtype=self.dtype)
        self.dones = torch.tensor([step[2] for step in steps], dtype=self.dtype)
        for i, done in enumerate(self.dones):
            if not done:
                continue
            self.observations[i] = torch.tensor(self.envs[i].reset())
            #TODO do we also need to reset the done?
        return observations, self.rewards, self.dones

