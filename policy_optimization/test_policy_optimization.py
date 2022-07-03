import numpy as np
import gym
import gym.spaces

from utils import repeat
from env_batch import EnvBatch
from models import SimpleLstmActor, SimpleLstmCritic
from ppo import Ppo
#from test_env import TestEnv


n_envs = 3
envs = EnvBatch(repeat(gym.make, 3, "CartPole-v1"))
obs_shape = envs.observation_space.shape
action_space = envs.action_space
print(f'O shape: {obs_shape}, A space: {action_space}')
assert isinstance(action_space, gym.spaces.Discrete), f'Non-single-discrete action space'
actor = SimpleLstmActor(observation_shape=obs_shape, n_actions=action_space.n)
critic = SimpleLstmCritic(observation_shape=obs_shape)
ppo = Ppo(
    envs=envs,
    actor=actor,
    critic=critic,
    ppo_clipping=0.1, # h&s use 0.2
    entropy_coefficient=0.01, # h&s use 0.1
    value_coefficient=0.5, # h&s use ?
    max_grad_norm=0.5, # h&s use 5
    learning_rate=2.5e-4, # h&s use 3e-4
    epochs_per_step=4,
    observation_shape=obs_shape,
    bptt_truncation_length=8,
    gae_horizon=128,
    buffer_size=3*64,
    minibatch_size=3*16,
    gamma=0.99, # h&s use 0.998
    gae_lambda=0.95, # h&s use 0.95
)
test_env = gym.make("CartPole-v1")
for i in range(10):
    ppo.train(10)
    ppo.test(test_env, episodes=3)
