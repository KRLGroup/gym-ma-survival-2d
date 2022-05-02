import gym
from gym import spaces

import multiagent_survival.engine as engine
import multiagent_survival.rendering as rendering


class MultiagentSurvivalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # [none, up, down, left, right]
    action_forces = [ (0., 0.), (0., 1.), (0., -1.), (-1., 0.), (1., 0.) ]

    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=float('-inf'), high=float('inf'), shape=(2,))
        self._world = None
        self._agent_body_id = None
        self._pygame = None
        self._window = None
        self._window_size = 512
        self._clock = None

    def reset(self, seed=None, return_info=False):
        super().reset(seed=seed)
        self._world = engine.World()
        self._agent_body_id = self._world.add_dynamic_body(position=(0., 0.))
        observation = self._world[self._agent_body_id].position
        info = {}
        return (observation, info) if return_info else observation

    def step(self, action):
        f = self.action_forces[action]
        self._world.apply_force((100*f[0], 100*f[1]), self._agent_body_id)
        self._world.step()
        observation = self._world[self._agent_body_id].position
        reward = 0.
        done = False
        info = {}
        return observation, reward, done, info

    def render(self, mode="human"):
        import pygame
        self._pygame = pygame
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if mode == 'human':
            self._init_human_rendering()
        #TODO maybe optimize by storing the surface between calls?
        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))
        rendering.draw_world(self._world, canvas)
        if mode == 'human':
            self._render_human(canvas)
        elif mode == 'rgb_array':
            transposed_img = np.array(pygame.surfarray.pixels3d(canvas))
            return np.transpose(transposed_img, axes=(1, 0, 2))
        else:
            assert False, 'How did we get here?'

    def _init_human_rendering(self):
        if self._window is None:
            self._pygame.init()
            self._pygame.display.init()
            self._window = self._pygame.display.set_mode(
                (self._window_size, self._window_size))
        if self._clock is None:
            self._clock = self._pygame.time.Clock()

    def _render_human(self, canvas):
        self._window.blit(canvas, canvas.get_rect())
        self._pygame.event.pump()
        self._pygame.display.update()
        self._clock.tick(self.metadata["render_fps"])        

    def close(self):
        import sys
        if self._pygame is None:
            return
        else:
            self._pygame.display.quit()
            self._pygame.quit()

