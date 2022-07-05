import numpy as np
import gym
import gym.spaces
import torch

from utils import repeat
from env_batch import EnvBatch
from ma_env import MaEnv
from models import SimpleLstmHiveMind, SimpleLstmHiveMindCritic
from ma_ppo_centralized import MaPpoCentralized
#from test_env import TestEnv


# assumes action space has only 1 dim and actor needs LSTM reset
def test_ma_episode(env: MaEnv, actor, render=False):
    actor.set_lstm_state(actor.zero_lstm_state(1, env.n_agents))
    done = True
    def inflate_observations(o):
        return torch.as_tensor(o).unsqueeze(0).unsqueeze(0)
    # shape: (1, 1, A, X)
    observations = env.reset()
    rs_list = []
    while True:
        if render:
            env.render()
        # as_ shape: (1, 1, A,)
        as_, _1, _2 = actor(inflate_observations(observations), torch.as_tensor(float(done)).view(1, -1), deterministic=True)
        # shapes (A, X), (A,), 1
        observations, rewards, done = env.step(as_.view(-1).numpy())
        rs_list.append(rewards)
        if done:
            break
    return np.array(rs_list)


n_envs = 3
n_agents = 2
envs = EnvBatch(repeat(lambda: MaEnv(repeat(gym.make, n_agents, "CartPole-v1")), n_envs))
n_observations=envs.envs[0].n_observations
n_actions=envs.envs[0].n_actions
actor = SimpleLstmHiveMind(n_observations, n_actions)
critic = SimpleLstmHiveMindCritic(n_observations)

ppo = MaPpoCentralized(
    envs=envs,
    actor=actor,
    critic=critic,
    ppo_clipping=0.1, # h&s use 0.2
    entropy_coefficient=0.01, # h&s use 0.1
    value_coefficient=0.5, # h&s use ?
    max_grad_norm=0.5, # h&s use 5
    learning_rate=2.5e-4, # h&s use 3e-4
    epochs_per_step=4,
    observation_shape=(n_observations,),
    bptt_truncation_length=8,
    gae_horizon=128,
    buffer_size=3*64,
    minibatch_size=3*16,
    gamma=0.99, # h&s use 0.998
    gae_lambda=0.95, # h&s use 0.95
)

test_env = MaEnv(repeat(gym.make, n_agents, "CartPole-v1"))

for i in range(10):
    ppo.train(10)
    rewards = test_ma_episode(test_env, actor, render=True)
    print(f'R = {rewards.sum(axis=0)}, r = {rewards.mean(axis=0)} +- {rewards.std(axis=0)}')
