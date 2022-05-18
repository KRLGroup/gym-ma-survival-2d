import time

import gym

from mas.envs.mas_env import MasEnv

def main(render: bool):
    env = MasEnv()
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    if render:
        env.render(mode='human')
    done = False
    start = time.time()
    for _ in range(1000):
        action = env.action_space.sample()
        #print(f'action = {action}')
        observation, reward, done, info = env.step(action)
        if render:
            env.render(mode='human')
        #print(f'observation = {observation}')
    end = time.time()
    print(f'avg step time: {(end - start)/1000.}')
    print(f'done')
    env.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        raise ValueError('Too many command line arguments')
    render = len(sys.argv) == 2 and sys.argv[1] == '--render'
    main(render=render)



