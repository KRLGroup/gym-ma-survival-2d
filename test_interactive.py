import numpy as np
import gym
import pygame

from mas.envs.mas_env import MasEnv

_n_controls = 4

def get_action_from_keyboard():
    keyboard = pygame.key.get_pressed()
    action = [0, 0]
    if keyboard[pygame.K_LEFT] and not keyboard[pygame.K_RIGHT]:
        action[0] = 2
    if keyboard[pygame.K_RIGHT] and not keyboard[pygame.K_LEFT]:
        action[0] = 1
    if keyboard[pygame.K_UP] and not keyboard[pygame.K_DOWN]:
        action[1] = 1
    if keyboard[pygame.K_DOWN] and not keyboard[pygame.K_UP]:
        action[1] = 2
    return action


def main():
    env = MasEnv()
    print(f'Controls: UP, DOWN, LEFT, RIGHT')
    env.reset()
    env.render(mode='human')
    done = False
    while not done:
        action = get_action_from_keyboard()
        #print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        #print(f'observation = {observation}')
    print(f'done')
    env.close()




if __name__ == '__main__':
    main()



