import gym
import pygame

from multiagent_survival.envs.multiagent_survival \
    import MultiagentSurvivalEnv

def get_action_from_keyboard():
    events = pygame.event.get()
    action = 0
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 1
            if event.key == pygame.K_DOWN:
                action = 2
            if event.key == pygame.K_LEFT:
                action = 3
            if event.key == pygame.K_RIGHT:
                action = 4
    return action


def main():
    env = MultiagentSurvivalEnv()
    print(f'Controls: UP, DOWN, LEFT, RIGHT')
    env.reset()
    action = 0
    done = False
    while not done:
        print(f'action: {action}')
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        print(f'observation = {observation}')
        action = get_action_from_keyboard()
    print(f'done')
    env.close()




if __name__ == '__main__':
    main()



