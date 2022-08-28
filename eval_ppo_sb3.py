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
from policy_optimization.vec_env_costume import VecEnvCostume

def unbatch_obs(obs, n_agents):
    xs = [dict() for _ in range(n_agents)]
    for k, v in obs.items():
        for i, x in enumerate(v):
            xs[i][k] = x
    return xs

exp_name = 'models/tmp'

env = VecEnvCostume(MaSurvivalEnv())

model = PPO.load(exp_name, env=env)

obs = env.reset()
done = False
R = np.zeros(env.env.n_agents)
frames = []
i = 0
while not done:
    frame = env.render()
    if i % 5 == 0:
        frames.append(frame)
    unbatched_obs = unbatch_obs(obs, env.env.n_agents)
    actions = np.vstack([model.predict(x)[0] for x in unbatched_obs])
    obs, rewards, dones, info = env.step(actions)
    done = dones[0]
    R += rewards
    i += 1
env.close()
print(f'R = {R}')

import imageio
from pygifsicle import optimize # type: ignore
gif_fpath = 'tmp.gif'
with imageio.get_writer(gif_fpath, mode='I') as writer:
    for frame in frames:
        writer.append_data(frame)
optimize(gif_fpath)