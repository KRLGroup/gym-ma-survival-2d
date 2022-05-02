#TODO typing stuff

import gym
from gym import spaces

from multiagent_survival.gym_hidenseek.envs.hide_and_seek \
    import HideAndSeek15x15Env
# from multiagent_survival.gym_hidenseek.envs.randomized_hide_and_seek \
#     import RandomizedHideAndSeek15x15Env
# from multiagent_survival.gym_hidenseek.envs.json_hide_and_seek \
#     import JsonHideAndSeek15x15Env
# from multiagent_survival.gym_hidenseek.envs.lock_and_return \
#     import LockAndReturn15x15Env
# from multiagent_survival.gym_hidenseek.envs.sequential_lock \
#     import SequentialLock15x15Env

class MultiagentSurvivalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    _color_map = {
      'agent': (255, 0, 0),
      } 

    def __init__(self, gym_hidenseek_env=None, *args, **kwargs):
        self.env = gym_hidenseek_env or HideAndSeek15x15Env(*args, **kwargs)
        #TODO don't hardcode this?
        self.action_space = spaces.MultiDiscrete([8,8,8,8])
        self.observation_space = NotImplemented
        self.done = False
        self._action_strings = ['forward', 'backward', 'clockwise',
                                'counterclockwise', 'lock', 'unlock', 'hold',
                                'release']
        self._window = None
        self._window_size = 512
        self._clock = None

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    # seed and return_info support is delegated to underlying env
    def reset(self, seed=None, return_info=False):
        super().reset(seed=seed)
        self.done = False
        return self.env.reset()
        #observation = self._get_obs()
        #info = self._get_info()
        #return (observation, info) if return_info else observation

    # aborts subsequent actions if one action terminates the episode
    def step(self, actions):
        agents = self.env.grid.getAgentList()
        assert (len(actions) == len(agents)), f'expected {len(agents)} actions, but {len(actions)} were given'
        observations = []
        rewards = []
        infos = []
        for action, agent in zip(actions, agents):
            if self.done:
                break
            action_string = self._action_strings[action]
            z, r, self.done, info = self.env.step(agent, action_string)
            observations.append(z)
            rewards.append(r)
            infos.append(info)
        #observation = self._get_obs()
        #info = self._get_info()
        return observations, rewards, self.done, infos

    # needs pygame to run
    def render(self, mode="human"):
        import pygame
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        
        if self._window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((self._window_size, self._window_size))
        if self._clock is None and mode == "human":
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))
        
        #TODO turn world coordinates into screen coordinates
        
        # pseudocode of the gym_hidenseek rendering:
        # - draw static stuff at init:
        #   - draw the floor
        #   - draw the grid
        #   - draw walls
        #   - draw cylinders
        # - draw dynamic, agent related stuff:
        #   - draw agents:
        #     - body polygon (color coded),
        #     - arrow
        #     - observation wedge
        #     - lidar circle
        #   - draw boxes:
        #     - polygon (color coded for locked etc. states)
        #     - cross for locked box
        #     - arrow
        #   - draw ramps:
        #     - polygon (color coded)
        #     - arrow
        
        for agent in self.env.grid.getAgentList():
            print(f'{[agent.vertex1, agent.vertex2, agent.vertex3, agent.vertex4]}')
            pygame.draw.polygon(
                surface=canvas, color=self._color_map['agent'],
                points=[agent.vertex1, agent.vertex2, agent.vertex3,
                        agent.vertex4])


        if mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        else:
            assert False, 'How did we get here?'

    def close(self):
        import sys
        if 'pygame' not in sys.modules:
            return
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()

