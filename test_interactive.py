import numpy as np
import gym
import pygame

from mas.envs.mas_env import MultiagentSurvivalEnv

_n_controls = 4

def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    action = 0
    pressed = np.array([keyboard[pygame.K_UP], keyboard[pygame.K_DOWN],
                        keyboard[pygame.K_LEFT], keyboard[pygame.K_RIGHT]],
                        dtype=bool)
    new = ~last_pressed & pressed
    if np.any(new):
        action = np.nonzero(new)[0][0] + 1
    elif np.any(pressed):
        action = np.nonzero(pressed)[0][0] + 1
    return action, pressed


def main():
    env = MultiagentSurvivalEnv()
    print(f'Controls: UP, DOWN, LEFT, RIGHT')
    env.reset()
    action = 0
    last_pressed = np.zeros(_n_controls, dtype=bool)
    done = False
    while not done:
        print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        print(f'observation = {observation}')
        action, last_pressed = get_action_from_keyboard(last_pressed)
    print(f'done')
    env.close()




if __name__ == '__main__':
    main()



