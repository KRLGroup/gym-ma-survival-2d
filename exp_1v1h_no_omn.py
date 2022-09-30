# NOTE: this code and related results rely on a prev version of the env OneVsOneHeals, in which the melee was continuous

from typing import Optional, List
import time
import os

import pygame

import numpy as np
import gym
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from masurvival.envs.masurvival_env import OneVsOne
from policy_optimization.vec_env_costume import VecEnvCostume
from policy_optimization.sb3_models import MhSaExtractor
from policy_optimization.sb3_log_multirewards import LogMultiRewards


def main():
    import sys
    model_dir = './models/'
    exp_name = '1v1h_no_omn'
    model_name = model_dir + exp_name + '/model'
    #TODO use os.path
    env = make_env()
    if len(sys.argv) == 1:
        print('error: expected subcommand from: train, eval, test_interactive, test_random')
        raise SystemExit(1)
    if sys.argv[1] == 'train':
        assert len(sys.argv) == 2
        env = VecEnvCostume(env)
        main_train(env, model_dir, exp_name, model_name)
    elif sys.argv[1] == 'eval':
        assert len(sys.argv) == 2
        env = VecEnvCostume(env)
        main_eval(env, model_dir, exp_name, model_name)
    elif sys.argv[1] == 'test_interactive':
        main_test_interactive(env, sys.argv[2:], model_dir + exp_name)
    elif sys.argv[1] == 'test_random':
        main_test_random_policy(env, sys.argv[2:], model_dir + exp_name)
    else:
        print(f'error: unrecognized subcommand: {sys.argv[1]}')


# build the env variant

config = {
    'observation': {
        'omniscent': False,
    },
    'reward_scheme': {
        'r_alive': 1,
        'r_dead': 0,
    },
    'gameover' : {
        'mode': 'alldead', # in {alldead, lastalive}
    },
    'rng': { 'seed': 42 },
    'spawn_grid': {
        'grid_size': 8,
        'floor_size': 25,
    },
    'agents': {
        'n_agents': 2,
        'agent_size': 1,
    },
    'health': {
        'health': 100,
    },
    'melee': {
        'range': 2,
        'damage': 35,
        'cooldown': 40, # makes sense as long as its greater than damage
        'drift': True, # so that we can actually render actions
    },
    'heals': {
        'reset_spawns': {
            'n_items': 15,
            'item_size': 0.5,
        },
        'heal': {
            'healing': 75,
        },
    },
    'safe_zone': {
        'phases': 8,
        'cooldown': 100,
        'damage': 1,
        'radiuses': [12.5, 10, 7.5, 5, 4, 2, 1],
        'centers': 'random',
    },
}

def make_env():
    return OneVsOne(config=config)


# train models

def main_train(env, model_dir, exp_name, model_name):
    try:
        model = PPO.load(model_name, env=env)
    except FileNotFoundError:
        print('No model file found, creating new model.')
        n_heals = env.env.config['heals']['reset_spawns']['n_spawns']
        entity_keys = {'zone', 'heals'}
        policy_kwargs = dict(
            features_extractor_class=MhSaExtractor,
            features_extractor_kwargs=dict(
                entity_keys=entity_keys,
                omniscent=env.env.config['observation']['omniscent'],
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
        # save freq is doubled for some sb3 reason
        save_freq=500_000, save_path=model_dir + exp_name + '/checkpoints',
        name_prefix='rl_model'
    )
    
    try:
        model.learn(
            total_timesteps=10_000_000,
            reset_num_timesteps=True,
            callback=[log_multirewards, checkpoint_callback],
        )
    except KeyboardInterrupt:
        pass
    
    model.save(model_name)


# eval trained models

def unbatch_obs(obs, n_agents):
    xs = [dict() for _ in range(n_agents)]
    for k, v in obs.items():
        for i, x in enumerate(v):
            xs[i][k] = x
    return xs

def main_eval(env, model_dir, exp_name, model_name):
    model = PPO.load(model_name, env=env)
    
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
    gif_fpath = model_dir + 'eval.gif'
    with imageio.get_writer(gif_fpath, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)
    optimize(gif_fpath)


# testing interactively

def zero_action():
    return [1, 1, 1, 0, 0, 0]

def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    #TODO update this with correct null actions
    action = zero_action()
    if keyboard[pygame.K_w] and not keyboard[pygame.K_s]:
        action[0] = 2
    if keyboard[pygame.K_s] and not keyboard[pygame.K_w]:
        action[0] = 0
    if keyboard[pygame.K_a] and not keyboard[pygame.K_d]:
        action[1] = 2
    if keyboard[pygame.K_d] and not keyboard[pygame.K_a]:
        action[1] = 0
    if keyboard[pygame.K_LEFT] and not keyboard[pygame.K_RIGHT]:
        action[2] = 2
    if keyboard[pygame.K_RIGHT] and not keyboard[pygame.K_LEFT]:
        action[2] = 0
    if keyboard[pygame.K_c] and not last_pressed[pygame.K_c]:
        action[3] = 1
    if keyboard[pygame.K_e] and not last_pressed[pygame.K_e]:
        action[4] = 1
    if keyboard[pygame.K_q] and not last_pressed[pygame.K_q]:
        action[5] = 1
    return action, keyboard

controls_doc = """Keyboard controls:
W,A,S,D: parallel and normal movement
LEFT,RIGHT: angular movement
C: attack
E: use item (last picked up)
Q: give item (last picked up)
-------------
"""

def test_interactive(
    env, gif_fpath: Optional[str] = None, record_interval: int = 10,
):
    print(controls_doc)
    env.reset()
    frames: List[np.ndarray] = []
    frame = env.render(mode='human')
    if gif_fpath is not None:
        frames.append(frame)
    done = False
    pressed = pygame.key.get_pressed()
    rewards = []
    i = 0
    while not done:
        actions = env.action_space.sample()
        user_action, pressed = get_action_from_keyboard(pressed)
        if env.n_agents >= 2:
            actions = (user_action,) + (zero_action(),) + actions[2:]
        elif env.n_agents == 1:
            actions = (user_action,)
        action = actions
        #print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        i += 1
        frame = env.render(mode='human')
        if gif_fpath is not None and (i+1) % record_interval == 0:
            frames.append(frame)
        #print(f'observation = {observation}')
        rewards.append(reward)
        if done:
            print(f'done after {i} steps')
            break
            obs = env.reset()
    env.close()
    rewards = np.array(rewards)
    print(f'R = {rewards.sum(axis=0)}, r ~ {rewards.mean(axis=0)} +- {rewards.std(axis=0)}')
    if gif_fpath is not None:
        import imageio
        from pygifsicle import optimize # type: ignore
        with imageio.get_writer(gif_fpath, mode='I') as writer:
            for frame in frames:
                writer.append_data(frame)
        optimize(gif_fpath)

def main_test_interactive(env, argv, exp_dir):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record-gif", dest='gif_fpath',
      type=str, default=None,
      help="record a GIF to the given file")
    parser.add_argument("--record-interval", dest='record_interval',
      type=int, default=10,
      help="only record frames each N environment steps")
    args = parser.parse_args(args=argv)
    gif_fpath = args.gif_fpath
    if gif_fpath is not None:
        gif_fpath = os.path.join(exp_dir, gif_fpath)
    test_interactive(
        env, gif_fpath=gif_fpath, record_interval=args.record_interval,
    )



# test env with a random policy

def test_random_policy(
    env, render_mode: Optional[str], gif_fpath: Optional[str] = None,
    record_interval: int = 10
):
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    frames: List[np.ndarray] = []
    if render_mode is not None:
        frame = env.render(mode=render_mode)
        if gif_fpath is not None:
            frames.append(frame)
    done = False
    start = time.time()
    i = 0
    while not done:
        action = env.action_space.sample()
        #print(f'action = {action}')
        observation, reward, done, info = env.step(action)
        i += 1
        if render_mode is not None:
            frame = env.render(mode=render_mode)
            if gif_fpath is not None and (i+1) % record_interval == 0:
                frames.append(frame)
        print(f'observation = {observation["heals_mask"]}')
        #print(reward)
        if done:
            print(f'done after {i} steps')
            break
            obs = env.reset()
    end = time.time()
    print(f'avg step time: {(end - start)/i}')
    print(f'done')
    env.close()
    if gif_fpath is not None:
        import imageio
        from pygifsicle import optimize # type: ignore
        with imageio.get_writer(gif_fpath, mode='I') as writer:
            for frame in frames:
                writer.append_data(frame)
        optimize(gif_fpath)

def main_test_random_policy(env, argv, exp_dir):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--render-mode", dest='render_mode',
      type=str, choices=['human', 'rgb_array'], default=None,
      help="render mode to use")
    parser.add_argument("-r", "--record-gif", dest='gif_fpath',
      type=str, default=None,
      help="record a GIF to the given file")
    parser.add_argument("--record-interval", dest='record_interval',
      type=int, default=10,
      help="only record frames each N environment steps")
    args = parser.parse_args(args=argv)
    gif_fpath = args.gif_fpath
    if gif_fpath is not None:
        gif_fpath = os.path.join(exp_dir, gif_fpath)
    test_random_policy(
        env, render_mode=args.render_mode,
        gif_fpath=gif_fpath,
        record_interval=args.record_interval
    )



if __name__ == '__main__':
    main()

# TODO:
# - [v] make melees discrete with cooldown
# - [x] move other agent obs to entities obs
# - [v] add obs for next safe zone
# - [x] use LSTMs
# - [x] add boxes of randomized shape? maybe not

# possible ideas:
# - make episodes longer and the map and zones bigger
# - give agents the obs on the next safe zone

# Tensorboard runs:
# PPO_180, PP0_181, PPO_182
