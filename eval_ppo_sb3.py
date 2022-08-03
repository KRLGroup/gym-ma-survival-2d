from typing import Optional, List
import time

import numpy as np
import gym
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO

from masurvival.envs.masurvival_env import MaSurvivalEnv

exp_name = 'models/ppo-mas-4heals-earlystop-2.5M.bak'

env = MaSurvivalEnv()

model = PPO.load(exp_name, env=env)

obs = env.reset()
done = False
R = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    R += reward
    env.render()
print(f'R = {R}')
