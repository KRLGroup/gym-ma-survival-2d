from typing import Optional, List
import time

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
from stable_baselines3.common.utils import obs_as_tensor

from torchinfo import summary

from masurvival.envs.masurvival_env import MaSurvivalEnv

from policy_optimization.sb3_models import MhSaExtractor


env = MaSurvivalEnv()

n_heals = env.config['heals']['reset_spawns']['n_spawns']
entity_keys = {'heals'}

policy_kwargs = dict(
    features_extractor_class=MhSaExtractor, 
    features_extractor_kwargs=dict(entity_keys=entity_keys)
)

model = PPO(MultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log='runs', n_steps=2048*8, batch_size=64*8, n_epochs=10*8)

obs = env.reset()
obs_tensor = obs_as_tensor(obs, None)
obs_tensor = { key: x.unsqueeze(0) for key, x in obs_tensor.items()}

summary(model.policy, input_data={'obs': obs_tensor}, col_names=("input_size", "output_size", "num_params"))
