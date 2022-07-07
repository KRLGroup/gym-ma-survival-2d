from typing import Optional, List
import time

import numpy as np
import gym

from masurvival.envs.masurvival_env import MaSurvivalEnv

def main(render_mode: Optional[str], gif_fpath: Optional[str] = None,
         record_interval: int = 10):
    env = MaSurvivalEnv()
    #print(env.action_space)
    #print(env.observation_space)
    env.reset()
    frames: List[np.ndarray] = []
    if render_mode is not None:
        frame = env.render(mode=render_mode)
        if gif_fpath is not None:
            frames.append(frame)
    done = False
    start = time.time()
    for i in range(1000):
        action = env.action_space.sample()
        #print(f'action = {action}')
        observation, reward, done, info = env.step(action)
        if render_mode is not None:
            frame = env.render(mode=render_mode)
            if gif_fpath is not None and (i+1) % record_interval == 0:
                frames.append(frame)
        #print(f'observation = {observation}')
        if done:
            print(f'done after {i} steps')
            break
            obs = env.reset()
    end = time.time()
    print(f'avg step time: {(end - start)/1000.}')
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
    parser.add_argument("-m", "--render-mode", dest='render_mode',
      type=str, choices=['human', 'rgb_array'], default=None,
      help="render mode to use")
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
    main(render_mode=args.render_mode, gif_fpath=args.gif_fpath,
         record_interval=args.record_interval)
