from typing import Optional, List
import time

import numpy as np
import gym
import gym.spaces

from stable_baselines3 import PPO

from masurvival.envs.masurvival_env import MaSurvivalEnv


env = MaSurvivalEnv()

exp_name = "ppo_mas-self-zone"

try:
    model = PPO.load(exp_name, env=env)
except FileNotFoundError:
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log='runs', n_steps=2048*8, batch_size=64*8, n_epochs=10*8)

try:
    model.learn(total_timesteps=500000, reset_num_timesteps=True)
except KeyboardInterrupt:
    pass

model.save(exp_name)

#del model # remove to demonstrate saving and loading

obs = env.reset()
done = False
R = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    R += reward
    env.render()
print(f'R = {R}')
