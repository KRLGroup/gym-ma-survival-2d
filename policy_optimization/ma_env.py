from typing import Any, List, Union, Callable, Tuple, Optional, Dict

import numpy as np
import gym
import gym.spaces


def decorate_observation(x, i):
    return np.concatenate([[i], x], axis=-1)

# turns a sequence of envs into a single multi-agent env
class MaEnv:

    #TODO attr annotations

    #TODO support setting seeds for the first resets?

    def __init__(self, envs: List[gym.Env], dtype: np.dtype = np.float32):
        self.envs = envs
        self.dtype = dtype
        self.n_agents = len(envs)
        self.observation_space = gym.spaces.Tuple([env.observation_space for env in envs])
        self.action_space = gym.spaces.Tuple([env.action_space for env in envs])
        # Only make sense if all envs share the same spaces.
        self.n_observations = 1 + np.array(envs[0].observation_space.shape).prod()
        self.n_actions = envs[0].action_space.n

    def reset(self):
        observations = np.array([decorate_observation(env.reset(), i) for i, env in enumerate(self.envs)], dtype=self.dtype)
        return observations

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        steps = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        observations = np.array([decorate_observation(step[0], i) for i, step in enumerate(steps)], dtype=self.dtype)
        rewards = np.array([step[1] for step in steps], dtype=self.dtype)
        dones = np.array([step[2] for step in steps], dtype=self.dtype)
        done = np.any(dones)
        return observations, rewards, done

    def render(self, *args, **kwargs):
        for env in self.envs:
            env.render(*args, **kwargs)

