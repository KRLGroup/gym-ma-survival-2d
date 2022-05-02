import numpy as np
import gym

from multiagent_survival.envs.multiagent_survival \
    import MultiagentSurvivalEnv

import random

def main():
    env = MultiagentSurvivalEnv()
    print(env.action_space)
    env.reset()
    for step in range(0, 100):
        actions = env.action_space.sample()
        print(f'[{step}] actions: {actions}')
        observations, rewards, done, info = env.step(actions)
        env.render(mode='human')
        print(f'[{step}]: {len(observations)}, {len(rewards)}, {done}')
        if done:
            print(f'Done after {step+1} steps')
            break
    env.close()

if __name__ == '__main__':
    main()



