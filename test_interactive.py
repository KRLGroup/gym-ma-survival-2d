import numpy as np
import gym
import pygame

from masurvival.envs.masurvival_env import MaSurvivalEnv

_n_controls = 4

def get_action_from_keyboard():
    keyboard = pygame.key.get_pressed()
    action = [0, 0, 0, 0]
    if keyboard[pygame.K_LEFT] and not keyboard[pygame.K_RIGHT]:
        action[1] = 1
    if keyboard[pygame.K_RIGHT] and not keyboard[pygame.K_LEFT]:
        action[1] = 2
    if keyboard[pygame.K_UP] and not keyboard[pygame.K_DOWN]:
        action[0] = 1
    if keyboard[pygame.K_DOWN] and not keyboard[pygame.K_UP]:
        action[0] = 2
    if keyboard[pygame.K_a]:
        action[2] = 1
    if keyboard[pygame.K_s]:
        action[3] = 1
    if keyboard[pygame.K_d]:
        action[3] = 2
    return action


def main():
    env = MaSurvivalEnv()
    print(f'Controls: UP: fwd, DOWN: bwd, LEFT: ccw, RIGHT: cw, '
          f'A: grab, S: lock, D: unlock')
    env.reset()
    env.render(mode='human')
    done = False
    while not done:
        actions = list(env.action_space.sample())
        actions[0] = get_action_from_keyboard()
        actions = tuple(actions)
        action = actions
        #print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        #print(f'observation = {observation}')
    print(f'done')
    env.close()


if __name__ == '__main__':
    main()

