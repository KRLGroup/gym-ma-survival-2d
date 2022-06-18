from typing import Any, Optional, Union, Tuple, List, Dict, Set, Callable
from types import ModuleType
from operator import attrgetter
import math

import numpy as np
import gym
from gym import spaces

from Box2D import b2World, b2Body, b2Fixture, b2Joint, b2Vec2 # type: ignore

import masurvival.simulation as sim
import masurvival.worldgen as worldgen
from masurvival.semantics import Agents, Boxes


AgentObservation = List[sim.LaserScan]
Observation = Tuple[AgentObservation, ...]
AgentAction = Tuple[int, int, int]
Action = Tuple[AgentAction, ...]

ConfigParameter = Union[int, float, str]
Config = Dict[str, Dict[str, ConfigParameter]]

default_config: Config = {
    'rng': {
        'seed': 42,
    },
    'spawn_grid': {
        'floor_size': 20,
        'grid_size': 4,
    },
    'boxes': {
        'n_boxes': 4,
        'box_size': 1,
    },
    'agents': {
        'n_agents': 4,
        'agent_size': 1,
        'lidar_n_lasers': 8,
        'lidar_fov': 0.9*np.pi,
        'lidar_depth': 2,
        'motor_linear_impulse': 0.5,
        'motor_angular_impulse': 0.025,
    },
}


class MaSurvivalEnv(gym.Env):

    # See the default values for available params
    config: Dict[str, Any] = default_config
    # actions and observation spaces
    agent_action_space: spaces.Space = spaces.MultiDiscrete([3,3,3])
    action_space: spaces.Space # changes based on number of agents
    observation_space: spaces.Space = NotImplemented
    # rendering
    metadata: Dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    window_size: int = 512 # only affects the first render call
    canvas: Optional[Any] = None
    # internal state
    simulation: sim.Simulation
    agents: Agents
    boxes: Boxes

    def __init__(self, config: Optional[Config] = None):
        if config is not None:
            self.config = config
        #TODO why does mypy think this is a float? :(
        n_agents = self.get_param('agents', 'n_agents')
        actions_spaces = (self.agent_action_space,)*n_agents # type: ignore
        self.action_space = spaces.Tuple(actions_spaces)
        self.simulation = sim.Simulation()
        self.agents = Agents(**self.config['agents'])
        self.boxes = Boxes(**self.config['boxes'])
        self.simulation.add_module(self.agents)
        self.simulation.add_module(self.boxes)

    # if given, the seed takes precedence over the config seed
    def reset(self, seed: int = None, return_info: bool = False,
              options: Optional[Dict[Any, Any]] = None) \
            -> Union[Observation,
                     Tuple[Observation, Dict[Any, Any]]]:
        if seed is None and 'seed' in self.config['rng']:
            seed = self.config['rng']['seed']
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        spawn_grid = worldgen.SpawnGrid(**self.config['spawn_grid'])
        spawn_grid.shuffle(self.rng)
        self.agents.spawner = spawn_grid
        self.boxes.spawner = spawn_grid
        self.simulation.reset()
        obs = self.fetch_observations()
        info: Dict = {}
        return (obs, info) if return_info else obs # type: ignore

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        assert self.action_space.contains(action), "Invalid action."
        self.queue_actions(action)
        self.simulation.step()
        obs = self.fetch_observations()
        reward = 0.
        done = False
        info: Dict = {}
        return obs, reward, done, info # type: ignore

    # Ignores the render mode after the first call. TODO change that
    # Always returns the rendered frame, even with 'human' mode.
    def render(self, mode: str = 'human') -> np.ndarray:
        import masurvival.rendering as rendering
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if self.canvas is None:
            agents_artist = rendering.AgentsArtist(
                self.agents, **rendering.AgentsArtist.default_config)
            boxes_artist = rendering.BoxesArtist(
                self.boxes, **rendering.BoxesArtist.default_config)
            self.canvas = rendering.Canvas(
                width=self.window_size, height=self.window_size, 
                world_size=self.config['spawn_grid']['floor_size'], 
                render_mode=mode, fps=self.metadata['render_fps'],
                artists=[agents_artist, boxes_artist])
        self.canvas.clear()
        return self.canvas.render()

    def close(self) -> None:
        if self.canvas is not None:
            self.canvas.close()

    # Gets the value used for a configuration parameter, either from the 
    # curent config or from the default values when absent.
    def get_param(self, key: str, subkey: str) -> ConfigParameter:
        config = self.config.get(key, None) or default_config[key]
        param = config.get(subkey, None) or default_config[key][subkey]
        return param

    def fetch_observations(self) -> Observation:
        return tuple(lidar.observation for lidar in self.agents.sensors)
    
    def queue_actions(self, actions: Action):
        d = [0., 1., -1.]
        for motor, action in zip(self.agents.motors, actions):
            control = ((d[action[0]], d[action[1]]), d[action[2]])
            motor.control(*control)

