from typing import Optional
import time

import gym

from masurvival.envs.masurvival_env import MaSurvivalEnv

def main(render_mode: Optional[str]):
    env = MaSurvivalEnv()
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    if render_mode is not None:
        render_frame = env.render(mode=render_mode)
    done = False
    start = time.time()
    for _ in range(1000):
        action = env.action_space.sample()
        #print(f'action = {action}')
        observation, reward, done, info = env.step(action)
        if render_mode is not None:
            render_frame = env.render(mode=render_mode)
        #print(f'observation = {observation}')
    end = time.time()
    print(f'avg step time: {(end - start)/1000.}')
    print(f'done')
    env.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 3:
        raise ValueError('Too many command line arguments')
    def_render_mode = 'human'
    render_mode = None
    if len(sys.argv) > 1:
        if sys.argv[1] != '--render':
            raise ValueError(f'Invalid command line argument')
        render_mode = def_render_mode
    if len(sys.argv) > 2:
        render_mode = sys.argv[2]
    main(render_mode=render_mode)



