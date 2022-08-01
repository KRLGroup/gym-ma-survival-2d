import sys

import numpy as np
import gym
import gym.spaces
import torch

from masurvival.envs.masurvival_env import MaSurvivalEnv, _zero_element

from policy_optimization.utils import repeat, recursive_apply
from policy_optimization.env_batch import EnvBatch
from policy_optimization.ma_env import MaEnv
from policy_optimization.models import EgocentricHiveMind, EgocentricHiveMindCritic #SimpleLstmHiveMind, SimpleLstmHiveMindCritic
from policy_optimization.ma_ppo_centralized import MaPpoCentralized
#from policy_optimization.test_env import TestEnv


# assumes action space has only 1 dim and actor needs LSTM reset
def test_ma_episode(env, actor, render=False):
    actor.set_lstm_state(actor.zero_lstm_state(1, env.n_agents))
    done = True
    def inflate_observations(o):
        return recursive_apply(lambda x: torch.as_tensor(x).unsqueeze(0).unsqueeze(0), o)
    # shape: (1, 1, A, X)
    observations = env.reset()
    rs_list = []
    while True:
        if render:
            env.render()
        # as_ shape: (1, 1, A,)
        #as_, _1, _2 = actor(inflate_observations(observations)[0], torch.as_tensor(float(done)).view(1, -1), deterministic=True)
        as_, _1, _2 = actor(*inflate_observations(observations), torch.as_tensor(float(done)).view(1, -1), deterministic=True)
        # shapes (A, X), (A,), 1
        observations, rewards, done, _ = env.step(as_.squeeze(0).squeeze(0).numpy())
        rs_list.append(rewards)
        if done:
            break
    return np.array(rs_list)


n_envs = 4
#envs = EnvBatch(repeat(lambda: MaEnv(repeat(gym.make, n_agents, "CartPole-v1")), n_envs))
envs = EnvBatch(repeat(MaSurvivalEnv, n_envs), verbose=False)
n_agents = envs.envs[0].n_agents
# n_observations=envs.envs[0].n_observations
# n_actions=envs.envs[0].n_actions

action_shape = [s.n for s in envs.envs[0].agent_action_space]
actor = EgocentricHiveMind(
    action_space=action_shape,
    x_agent_size=envs.envs[0].observation_sizes[0],
    x_lidar_size=envs.envs[0].observation_sizes[1],
    x_safe_zone_size=envs.envs[0].observation_sizes[2],
    entity_types=len(envs.envs[0].observation_sizes[3])
)
critic = EgocentricHiveMindCritic(
    x_agent_size=envs.envs[0].observation_sizes[0],
    x_lidar_size=envs.envs[0].observation_sizes[1],
    x_safe_zone_size=envs.envs[0].observation_sizes[2],
    entity_types=len(envs.envs[0].observation_sizes[3])
)
# actor = SimpleLstmHiveMind(8, action_shape)
# critic = SimpleLstmHiveMindCritic(8)

#sys.exit()

ppo = MaPpoCentralized(
    envs=envs,
    observation_shapes=recursive_apply(lambda x: x.shape, _zero_element(envs.envs[0].observation_space)),
    action_size=len(action_shape),
    actor=actor,
    critic=critic,
    ppo_clipping=0.2, # h&s use 0.2
    entropy_coefficient=0.01,
    value_coefficient=0.5, # h&s use ?
    max_grad_norm=0.5, # h&s use 5
    learning_rate=3e-4, # h&s use 3e-4
    epochs_per_step=50,
    bptt_truncation_length=16,
    gae_horizon=128,
    buffer_size=128*128,
    minibatch_size=4096,
    gamma=0.99, # h&s use 0.998
    gae_lambda=0.95, # h&s use 0.95
    verbose=True
)

#test_env = MaEnv(repeat(gym.make, n_agents, "CartPole-v1"))

for i in range(100):
    ppo.train(1)
    print(f'Trained 1 time')
    for _ in range(3):
        test_env = MaSurvivalEnv()
        rewards = test_ma_episode(test_env, actor, render=True)
        print(f'R = {rewards.sum(axis=0)}, r = {rewards.mean(axis=0)} +- {rewards.std(axis=0)}')
        test_env.close()
