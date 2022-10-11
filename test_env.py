from typing import Optional, List
import time
import os
import json
import pprint

import pygame

import numpy as np
import gym
import gym.spaces

from masurvival.envs.masurvival_env import MaSurvival


class MultiAgentPolicy:
    def act(self, observations):
        raise NotImplementedError

class RandomPolicy(MultiAgentPolicy):
    def __init__(self, env):
        self.action_space = env.action_space
    def act(self, observations):
        return self.action_space.sample()

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

class HumanPolicy(MultiAgentPolicy):

    def __init__(self, env):
        self.action_space = env.action_space
        self.n_agents = env.n_agents

    def act(self, observations):
        actions = self.action_space.sample()
        user_action, self.pressed = get_action_from_keyboard(self.pressed)
        if self.n_agents >= 2:
            actions = (user_action,) + actions[1:]
        elif self.n_agents == 1:
            actions = (user_action,)
        return actions


def test_env(
    env,
    mode: str = 'random',
    render_mode: Optional[str] = None,
    record_interval: Optional[int] = None,
):
    interactive = mode == 'interactive'
    if interactive:
        print(controls_doc)
        render_mode = 'human'
        policy = HumanPolicy(env)
    elif mode == 'random':
        policy = RandomPolicy(env)
    else:
        assert False, 'How did we get here?'
    observation = env.reset()
    frames: List[np.ndarray] = []
    if render_mode is not None:
        frame = env.render(mode='human')
        if record_interval is not None:
            frames.append(frame)
    if interactive:
        policy.pressed = pygame.key.get_pressed()
    done = False
    start = time.time()
    while not done:
        action = policy.act(observation)
        observation, reward, done, info = env.step(action)
        i = env.stats['steps']
        if render_mode is not None:
            frame = env.render(mode='human')
            if record_interval is not None and (i+1) % record_interval == 0:
                frames.append(frame)
        #print(f'x = {observation}')
    end = time.time()
    print('Episode complete. Stats printed below.')
    if not interactive:
        print(f'avg env step time: {(end - start)/i} (s)')
    stats = env.flush_stats()
    env.close()
    pprint.PrettyPrinter().pprint(stats)
    if record_interval is not None:
        return frames



# Script stuff

def main(mode, env_config_fpath, render_mode, gif_fpath, record_interval):
    with open(env_config_fpath) as f:
        config = json.load(f)
    env = MaSurvival(config=config)
    if gif_fpath is None:
        record_interval = None
    maybe_frames = test_env(
        env,
        mode=mode,
        render_mode=render_mode,
        record_interval=record_interval
    )
    if gif_fpath is not None:
        assert maybe_frames is not None
        frames = maybe_frames
        import imageio
        from pygifsicle import optimize # type: ignore
        with imageio.get_writer(gif_fpath, mode='I') as writer:
            for frame in frames:
                writer.append_data(frame)
        optimize(gif_fpath)

argparse_desc = \
'Test the environment for one episode, either '
'interactively or with a random policy.'

argparse_args = [
    (['mode'], dict(
        metavar='MODE',
        type=str,
        choices=['interactive', 'random'],
        default='random',
        help='Which type of policy to use for testing.',
    )),
    (['-c', '--config'], dict(
        dest='env_config_fpath',
        metavar='PATH',
        type=str,
        default=None,
        help='Use the given JSON file as the env configuration.'
    )),
    (['-m', '--render-mode'], dict(
        dest='render_mode',
        type=str,
        choices=['human', 'rgb_array'],
        default=None,
        help='The render mode to use.',
    )),
    (['-r', '--record-gif'], dict(
        dest='gif_fpath',
        metavar='PATH',
        type=str,
        default=None,
        help='Record a GIF to the given file.'
    )),
    (['--record-interval'], dict(
        dest='record_interval',
        metavar='N',
        type=int,
        default=10,
        help='Only record frames each N environment steps.'
    )),
]

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=argparse_desc)
    for args, kwargs in argparse_args:
        argparser.add_argument(*args, **kwargs)
    cli_args = argparser.parse_args()
    #print(vars(cli_args))
    main(**vars(cli_args))

