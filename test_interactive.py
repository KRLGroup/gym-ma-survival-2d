import numpy as np
import gym
import pygame

from masurvival.envs.masurvival_env import MaSurvivalEnv

_n_controls = 4

def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    action = [0, 0, 0, 0, 0]
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
    if keyboard[pygame.K_LSHIFT]:
        action[3] = 1
    if keyboard[pygame.K_e] and not last_pressed[pygame.K_e]:
        action[4] = 1
    return action, keyboard

controls_doc = """Keyboard controls:
W,A,S,D: parallel and normal movement
LEFT,RIGHT: angular movement
SHIFT: hold closest valid object in range
E: lock/unlock closest valid object in range
-------------
"""

def main():
    env = MaSurvivalEnv(n_agents=4)
    print(controls_doc)
    env.reset()
    env.render(mode='human')
    done = False
    pressed = pygame.key.get_pressed()
    while not done:
        actions = list(env.action_space.sample())
        actions[0], pressed = get_action_from_keyboard(pressed)
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

