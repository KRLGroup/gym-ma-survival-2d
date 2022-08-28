from typing import Optional, List
import time

import numpy as np
import gym
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from masurvival.envs.masurvival_env import MaSurvivalEnv
from policy_optimization.vec_env_costume import VecEnvCostume
from policy_optimization.sb3_models import MhSaExtractor
from policy_optimization.sb3_log_multirewards import LogMultiRewards


exp_name = 'models/tmp'

env = VecEnvCostume(MaSurvivalEnv())

try:
    model = PPO.load(exp_name, env=env)
except FileNotFoundError:
    n_heals = env.env.config['heals']['reset_spawns']['n_spawns']
    entity_keys = {'heals'}
    policy_kwargs = dict(
        features_extractor_class=MhSaExtractor,
        features_extractor_kwargs=dict(
            entity_keys=entity_keys
        )
    )
    model = PPO(
        MultiInputActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='runs',
        n_steps=2048*8,
        batch_size=64*8,
        n_epochs=50*8,
        target_kl=0.01,
    )

log_multirewards = LogMultiRewards()
checkpoint_callback = CheckpointCallback(
    save_freq=500000, save_path='./models/checkpoints_tmp/',
    name_prefix='rl_model'
)

try:
    model.learn(
        total_timesteps=10000000,
        reset_num_timesteps=True,
        callback=[log_multirewards, checkpoint_callback],
    )
except KeyboardInterrupt:
    pass

model.save(exp_name)

# obs = env.reset()
# done = False
# R = 0
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     R += reward
#     env.render()
# print(f'R = {R}')