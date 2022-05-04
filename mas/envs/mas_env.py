from typing import Any, Optional, Union, Tuple, List, Dict, Callable
from types import ModuleType
from operator import attrgetter

import numpy as np
from numpy.typing import ArrayLike
import gym
from gym import spaces

from Box2D import b2World, b2Body # type: ignore

#TODO make this optional
import pygame

from mas.types import Vec2
import mas.simulation as simulation
import mas.worldgen as worldgen
import mas.rendering as rendering


class MultiagentSurvivalEnv(gym.Env):

    # simulation parameters
    simulation_substeps: int = 2
    velocity_iterations: int = 10
    position_iterations: int = 10
    # [0, 1, 2] -> [0., 1., -1.]
    _impulses: List[float] = [ 0., 1., -1. ]

    # gym spaces    
    action_space: spaces.Space = spaces.MultiDiscrete([3,3])
    observation_space: spaces.Space = spaces.Box(
        low=float('-inf'), high=float('inf'), shape=(2,2))
    
    # world state
    _world_size: float = 20.
    _world: b2World
    _agent: b2Body
    _box: b2Body
    _pillar: b2Body
    _walls: List[b2Body]
    
    # rendering
    metadata: Dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30}
    _colors: Dict[str, rendering.Color] = {
        'agent': (0, 0, 255),
        'box': (255, 0, 0)}
    _window: Optional[pygame.surface.Surface] = None
    _window_size: int = 512
    _clock: Optional[pygame.time.Clock] = None

    def __init__(self, simulation_substeps: int = 2,
                 velocity_iterations: int = 10,
                 position_iterations: int = 10) -> None:
        self.simulation_substeps = simulation_substeps
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    def reset(self, *, seed: int = None, return_info: bool = False,
              options: Optional[Dict[Any, Any]] = None) \
            -> Union[ArrayLike, Tuple[ArrayLike, Dict[Any, Any]]]:
        super().reset(seed=seed)
        self._world = b2World(gravity=(0, 0), doSleep=True)
        bodies = worldgen.populate_world(self._world, self._world_size)
        self._agent, self._box = bodies[0:2]
        observation = self._get_obs()
        info: Dict = {}
        return (observation, info) if return_info else observation

    def step(self, action: Tuple[int, int]) \
            -> Tuple[ArrayLike, float, bool, Dict]:
        impulse = (self._impulses[action[0]], self._impulses[action[1]])
        simulation.apply_impulse(impulse, self._agent)
        simulation.simulate(
            world=self._world, substeps=self.simulation_substeps, 
            velocity_iterations=self.velocity_iterations, 
            position_iterations=self.position_iterations)
        observation = self._get_obs()
        reward = 0.
        done = False
        info: Dict = {}
        return observation, reward, done, info

    def _get_obs(self) -> ArrayLike:
        agent_pos = self._agent.position
        box_pos = self._box.position
        observation = np.array([agent_pos, box_pos])
        return observation

    def render(self, mode: str = 'human') -> Optional[ArrayLike]:
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if mode == 'human':
            self._init_human_rendering()
        #TODO maybe optimize by storing the surface between calls?
        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))
        rendering.draw_world(canvas, self._world, self._world_size, 
                             self._colors)
        if mode == 'human':
            self._render_human(canvas)
        elif mode == 'rgb_array':
            transposed_img = np.array(pygame.surfarray.pixels3d(canvas))
            return np.transpose(transposed_img, axes=(1, 0, 2))
        else:
            assert False, 'How did we get here?'
        return None # make mypy happy :D

    def _init_human_rendering(self) -> None:
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._window_size, self._window_size))
        if self._clock is None:
            self._clock = pygame.time.Clock()

    def _render_human(self, canvas) -> None:
        assert(self._window is not None)
        assert(self._clock is not None)
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata["render_fps"])        

    def close(self) -> None:
        if self._window is None:
            return
        pygame.display.quit()
        pygame.quit()

