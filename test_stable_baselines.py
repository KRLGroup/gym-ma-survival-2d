from typing import Optional, List
import time

import numpy as np
import gym
import gym.spaces

from stable_baselines3 import PPO

from masurvival.envs.masurvival_env import MaSurvivalEnv


class SaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        R = dict(low=float('-inf'), high=float('inf'))
        z_size = env.observation_space[0].shape[-1]
        z_size += env.observation_space[1].shape[-1]
        z_size += env.observation_space[2].shape[-1]
        self._observation_space = gym.spaces.Box(**R, shape=(z_size,))
        self._action_space = env.action_space[0]
    def _sa_obs(self, z):
        z_agent = z[0][0]
        z_lidar = z[1][0]
        z_zone = z[2][0]
        z_sa = np.concatenate([z_agent, z_lidar, z_zone], axis=-1)
        return z_sa
    def reset(self):
        z = self.env.reset()
        return self._sa_obs(z)
    def step(self, action):
        z, r, d, info = self.env.step((action,))
        return self._sa_obs(z), r[0], d, info


# Parallel environments
env = SaWrapper(MaSurvivalEnv())
#print(env.observation_space)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='runs')
try:
  model.learn(total_timesteps=1000000)
except KeyboardInterrupt:
  pass
model.save("ppo_mas-heals-1000000")
#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo_cartpole")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
