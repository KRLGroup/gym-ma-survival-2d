from typing import Optional, List

import numpy as np
import gym
import pygame

from masurvival.envs.masurvival_env import MaSurvivalEnv

def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    action = [0, 0, 0, 0]
    if keyboard[pygame.K_w] and not keyboard[pygame.K_s]:
        action[0] = 1
    if keyboard[pygame.K_s] and not keyboard[pygame.K_w]:
        action[0] = 2
    if keyboard[pygame.K_a] and not keyboard[pygame.K_d]:
        action[1] = 1
    if keyboard[pygame.K_d] and not keyboard[pygame.K_a]:
        action[1] = 2
    if keyboard[pygame.K_LEFT] and not keyboard[pygame.K_RIGHT]:
        action[2] = 1
    if keyboard[pygame.K_RIGHT] and not keyboard[pygame.K_LEFT]:
        action[2] = 2
    if keyboard[pygame.K_e] and not last_pressed[pygame.K_e]:
        action[3] = 1
    return action, keyboard

controls_doc = """Keyboard controls:
W,A,S,D: parallel and normal movement
LEFT,RIGHT: angular movement
E: attack
-------------
"""

def main(gif_fpath: Optional[str] = None, record_interval: int = 10):
    env = MaSurvivalEnv()
    print(controls_doc)
    env.reset()
    frames: List[np.ndarray] = []
    frame = env.render(mode='human')
    if gif_fpath is not None:
        frames.append(frame)
    done = False
    pressed = pygame.key.get_pressed()
    for i in range(1000):
        actions = env.action_space.sample()
        user_action, pressed = get_action_from_keyboard(pressed)
        actions = (user_action,) + actions[1:]
        action = actions
        #print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        frame = env.render(mode='human')
        if gif_fpath is not None and (i+1) % record_interval == 0:
            frames.append(frame)
        #print(f'observation = {observation}')
    print(f'done')
    env.close()
    if gif_fpath is not None:
        import imageio
        from pygifsicle import optimize # type: ignore
        with imageio.get_writer(gif_fpath, mode='I') as writer:
            for frame in frames:
                writer.append_data(frame)
        optimize(gif_fpath)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record-gif", dest='gif_fpath',
      type=str, default=None,
      help="record a GIF to the given file")
    parser.add_argument("--record-interval", dest='record_interval',
      type=int, default=10,
      help="only record frames each N environment steps")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(gif_fpath=args.gif_fpath, record_interval=args.record_interval)

