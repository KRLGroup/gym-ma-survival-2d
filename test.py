from typing import Optional, List, Union
import time
import os
import json
import pprint

import pygame

import numpy as np
import gym
import gym.spaces

from stable_baselines3 import PPO
from sb3_utils.vec_env_costume import VecEnvCostume

from masurvival.envs.masurvival_env import MaSurvival

import experiments


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


def unbatch_obs(obs, n_agents):
    xs = [dict() for _ in range(n_agents)]
    for k, v in obs.items():
        for i, x in enumerate(v):
            xs[i][k] = x
    return xs

# assumes a vec env
class TrainedPolicy(MultiAgentPolicy):

    def __init__(self, env, model):
        self.env = env
        self.model = model

    def act(self, observations):
        unbatched_obs = unbatch_obs(observations, self.env.env.n_agents)
        actions = np.vstack([self.model.predict(x)[0] for x in unbatched_obs])
        return actions


def test_env(
    env,
    policy: Union[str, PPO] = 'random',
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_interval: Optional[int] = None,
    test_performance: bool = False,
):
    is_model = isinstance(policy, PPO)
    interactive = policy == 'interactive'
    if interactive:
        print(controls_doc)
        render_mode = 'human'
        policy = HumanPolicy(env)
    elif policy == 'random':
        policy = RandomPolicy(env)
    elif is_model:
        policy = TrainedPolicy(env, model=policy)
    else:
        assert False, 'How did we get here?'
    if not is_model:
        env.env = env # for convenience
    observation = env.reset()
    frames: List[np.ndarray] = []
    if render_mode is not None:
        frame = env.env.render(mode=render_mode)
        if record_interval is not None:
            frames.append(frame)
    if interactive:
        policy.pressed = pygame.key.get_pressed()
    if test_performance:
        performances = []
    done = False
    while not done:
        action = policy.act(observation)
        start = time.process_time()
        observation, reward, done, info = env.step(action)
        end = time.process_time()
        if test_performance:
            performances.append(end - start)
        if is_model:
            done = done[0]
        i = env.env.stats['steps']
        if render_mode is not None:
            frame = env.env.render(mode=render_mode)
            if record_interval is not None and (i+1) % record_interval == 0:
                frames.append(frame)
        #print(f'x = {observation}')
        if max_steps is not None and i >= max_steps:
            print(f'Maximum number of steps reached, terminating episode.')
            break
    print('Episode complete. Stats printed below.')
    stats = env.env.flush_stats()
    env.close()
    pprint.PrettyPrinter().pprint(stats)
    if test_performance:
        perfs = np.array(performances)
        print(f'Performance test results: {perfs.mean()}, {perfs.std()}')
    if record_interval is not None:
        return frames



# Script stuff

def main(
    policy,
    max_steps,
    checkpoint_id,
    env_config_fpath,
    render_mode,
    screenshot,
    gif_fpath,
    record_interval,
    test_performance,
):
    is_model = policy not in {'interactive', 'random'}
    if is_model:
        exp_dirpath = policy
        exp = experiments.get_experiment(exp_dirpath)
        run_id = experiments.get_run_ids(exp)[-1]
        run = experiments.get_run(exp, run_id)
        if checkpoint_id is not None:
            policy = experiments.get_checkpoint_model(run, checkpoint_id)
        else:
            policy = run['model']
        print(f'Loading environment configuration from {exp["name"]}.')
        config = exp['env_config']
    else:
        with open(env_config_fpath) as f:
            config = json.load(f)
    env = MaSurvival(config=config)
    if screenshot:
        gif_fpath = None
        render_mode = 'rgb_array'
        record_interval = 1
    elif gif_fpath is None:
        record_interval = None
    if test_performance:
        policy = 'random'
        is_model = False
    if is_model:
        env = VecEnvCostume(env)
        print(f'Loading saved policy model at {policy}.')
        policy = PPO.load(policy, env=env)
    maybe_frames = test_env(
        env,
        policy=policy,
        max_steps=max_steps,
        render_mode=render_mode,
        record_interval=record_interval,
        test_performance=test_performance,
    )
    if screenshot:
        import imageio
        if is_model:
            img_fpath = f'{exp["name"]}.png'
        else:
            img_fpath = f'{os.path.splitext(env_config_fpath)[0]}.png'
        print(f'Saving screenshot to {img_fpath}.')
        imageio.imsave(img_fpath, maybe_frames[-1])
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
    (['policy'], dict(
        metavar='POLICY',
        type=str,
        default='random',
        help='The policy to use for testing. If not "random" or "interactive", the string is treated as the path to an experiment.',
    )),
    (['--max-steps'], dict(
        dest='max_steps',
        metavar='STEPS',
        type=int,
        default=None,
        help='Run only for the given amount of steps.'
    )),
    (['-s', '--checkpoint'], dict(
        dest='checkpoint_id',
        metavar='STEPS',
        type=int,
        default=None,
        help='Use the checkpoint at the given time steps instead of the last model.'
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
    (['--screenshot'], dict(
        action='store_true',
        default=False,
        help='Run only the first step, recording a PNG image of the first frame of the episode.'
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
    (['--test-performance'], dict(
        action='store_true',
        default=False,
        help='Test the performance of the environment by using a random policy over 100 steps, reporting the mean and standard deviation of the time taken by the each env steps.'
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

