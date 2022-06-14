from typing import Any, Optional, Union, Tuple, List, Dict, Set, Callable
from types import ModuleType
from operator import attrgetter
import math

import numpy as np
import gym
from gym import spaces

from Box2D import b2World, b2Body, b2Fixture, b2Joint, b2Vec2 # type: ignore

#TODO factor this out into rendering module
import pygame

import masurvival.simulation as simulation
import masurvival.worldgen as worldgen
#TODO make this optional
import masurvival.rendering as rendering


AgentObservation = simulation.LidarScan
Observation = Tuple[AgentObservation, ...]
AgentAction = Tuple[int, int, int, int, int]

class MaSurvivalEnv(gym.Env):

    # simulation parameters
    simulation_substeps: int = 2
    velocity_iterations: int = 10
    position_iterations: int = 10    

    # worldgen parameters
    _n_agents: int = 4
    _n_ramps: int = 2
    _n_boxes: int = 2
    _n_pillars: int = 2

    # actions
    agent_action_space: spaces.Space = spaces.MultiDiscrete([3,3,3,2,2])
    action_space: spaces.Space
    # [0, 1, 2] -> [0., 1., -1.]
    _impulses: List[float] = [ 0., 1., -1. ]
    # coefficients for the linear and angular impulses
    _acc_sens: float = 0.5
    _turn_sens: float = 0.025
    # coefficient for boosting angular impulse when the agent is holding an 
    # object; this makes turning easier
    _hold_turn_sens_boost: float = 3.0
    _hold_relative_range: float = 0.05
    _lock_relative_range: float = 0.05
    _hold_range: float
    _lock_range: float
    
    # observations
    observation_space: spaces.Space = NotImplemented
    _n_lasers: int = 8
    _lidar_angle: float = 0.8*math.pi
    _lidar_relative_depth: float = 0.1
    _lidar_depth: float
    _obs: Observation
    
    # world state
    _world_size: float = 20.
    _world: b2World
    _agents: List[b2Body]
    _spawn_grid_xs: np.ndarray
    _spawn_grid_ys: np.ndarray
    _free_spawn_cells: Set[int]

    # RNG state    
    _rng: np.random.Generator
    
    # rendering
    metadata: Dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30}
    _colors: Dict[str, rendering.Color] = {
        'default': pygame.Color('gold'),
        'held': pygame.Color('sienna1'),
        'locked': pygame.Color('slateblue3'),
        'ground': pygame.Color('white'),
        'wall': pygame.Color('gray'),
        'pillar': pygame.Color('gray'),
        'agent': pygame.Color('cornflowerblue'),
        'ramp': pygame.Color('white'),
        'lidar_off': pygame.Color('gray'),
        'lidar_on': pygame.Color('indianred2'),
        'free_cell': pygame.Color('green'),
        'full_cell': pygame.Color('red'),
        'hand_on': pygame.Color('springgreen2'),
        'hand_off': pygame.Color('gray'),
        'ramp_edge': pygame.Color('red')}
    _outline_colors: Dict[str, rendering.Color] = {
        'default': pygame.Color('gray25'),}
    _window_size: int = 512
    _canvas: Optional[rendering.Canvas] = None

    def __init__(self, simulation_substeps: int = 2,
                 velocity_iterations: int = 10, position_iterations: int = 10,
                 n_agents: Optional[int] = None,
                 n_boxes: Optional[int] = None,
                 n_pillars: Optional[int] = None,
                 spawn_grid_size: int = 5,) -> None:
        self.simulation_substeps = simulation_substeps
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        if n_agents is not None:
            self._n_agents = n_agents
        if n_boxes is not None:
            self._n_boxes = n_boxes
        if n_pillars is not None:
            self._n_pillars = n_pillars
        self._spawn_grid_xs, self._spawn_grid_ys \
            = worldgen.uniform_grid(cells_per_side=spawn_grid_size, 
                                    grid_size=self._world_size)
        self._lidar_depth = self._lidar_relative_depth*self._world_size
        self._hold_range = self._hold_relative_range*self._world_size
        self._lock_range = self._lock_relative_range*self._world_size
        self.action_space \
            = spaces.Tuple((self.agent_action_space,)*self._n_agents)

    def reset(self, seed: int = None, return_info: bool = False,
              options: Optional[Dict[Any, Any]] = None) \
            -> Union[Observation,
                     Tuple[Observation, Dict[Any, Any]]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed=seed)
        self._world = simulation.empty_flatland()
        bodies, self._free_spawn_cells = worldgen.populate_world(
            self._world, self._world_size, 
            spawn_grid_xs=self._spawn_grid_xs, 
            spawn_grid_ys=self._spawn_grid_ys, n_agents=self._n_agents,
            n_ramps=self._n_ramps, n_boxes=self._n_boxes, 
            n_pillars=self._n_pillars, rng=self._rng)
        self._agents = bodies['agents']
        obs = [self._observe(i) for i in range(self._n_agents)]
        self._obs = tuple(obs)
        info: Dict = {}
        return (self._obs, info) if return_info else self._obs

    def step(self, action: Tuple[AgentAction, ...]) \
            -> Tuple[Observation, float, bool, Dict]:
        assert self.action_space.contains(action), "Invalid action."
        for i, agent in enumerate(self._agents):
            self._act(agent, action[i])
        simulation.simulate(
            world=self._world, substeps=self.simulation_substeps, 
            velocity_iterations=self.velocity_iterations, 
            position_iterations=self.position_iterations)
        obs = [self._observe(i) for i in range(self._n_agents)]
        self._obs = tuple(obs)
        reward = 0.
        done = False
        info: Dict = {}
        return self._obs, reward, done, info

    def _act(self, agent: b2Body, action: AgentAction) -> None:
        self._perform_movement(agent, action[0], action[1], action[2])
        self._perform_hold(agent, action[3])
        self._perform_lock(agent, action[4])

    def _perform_movement(
            self, agent: b2Body, parallel_action: int, normal_action: int, 
            angular_action: int):
        R = agent.transform.R
        x, y = b2Vec2(1., 0.), b2Vec2(0., 1.)
        parallel_impulse = self._acc_sens*self._impulses[parallel_action]
        normal_impulse = self._acc_sens*self._impulses[normal_action]
        angular_impulse = self._turn_sens*self._impulses[angular_action]
        if 'holds' in agent.userData:
            angular_impulse *= self._hold_turn_sens_boost
        simulation.apply_impulse(parallel_impulse*(R*x), agent)
        simulation.apply_impulse(normal_impulse*(R*y), agent)
        simulation.apply_angular_impulse(angular_impulse, agent)        

    def _perform_hold(self, agent: b2Body, hold_action: int):
        if 'holds' in agent.userData and hold_action == 1:
            return
        if 'holds' not in agent.userData and hold_action == 0:
            return
        if hold_action == 0:
            held_body, hold_joint = agent.userData['holds']
            self._world.DestroyJoint(hold_joint)
            del held_body.userData['heldBy']
            del agent.userData['holds']
            return
        assert hold_action == 1, "How did we get here?"
        hold_scan = simulation.laser_scan(
            world=self._world, transform=agent.transform, angle=0., 
            depth=self._hold_range)
        if hold_scan is None:
            return
        body = hold_scan[0].body
        if not body.userData.get('holdable', False):
            return
        if 'heldBy' in body.userData or 'lockedBy' in body.userData:
            return
        hold_joint = simulation.holding_joint(
            holder=agent, held=body, world=self._world)
        body.userData['heldBy'] = agent
        agent.userData['holds'] = body, hold_joint

    def _perform_lock(self, agent: b2Body, lock_action: int):
        if lock_action == 0:
            return
        assert lock_action == 1, 'How did we get here?'
        lock_scan = simulation.laser_scan(
            world=self._world, transform=agent.transform, angle=0., 
            depth=self._lock_range)
        if lock_scan is None:
            return
        body = lock_scan[0].body
        if not body.userData.get('lockable', False):
            return
        if 'heldBy' in body.userData:
            return
        lockedBy = body.userData.get('lockedBy', None)
        if lockedBy is not None and lockedBy is not agent:
            return
        if lockedBy is None:
            simulation.set_static(body)
            body.userData['lockedBy'] = agent
            agent.userData['locks'] = body
        else: # lockedBy is agent
            simulation.set_dynamic(body)
            del body.userData['lockedBy']
            del agent.userData['locks']

    def _observe(self, agent_id: int) -> AgentObservation:
        agent = self._agents[agent_id]
        lidar_scan = simulation.lidar_scan(
            world=self._world, n_lasers=self._n_lasers, 
            transform=agent.transform, angle=self._lidar_angle, 
            radius=self._lidar_depth)
        return lidar_scan

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if self._canvas is None:
            self._canvas = rendering.Canvas(
                width=self._window_size, height=self._window_size, 
                world_size=self._world_size, 
                background=self._colors['ground'], render_mode=mode, 
                surfaces=16, fps=self.metadata['render_fps'])
        self._canvas.clear()
        for i in range(self._spawn_grid_xs.shape[0]):
            x, y = self._spawn_grid_xs[i], self._spawn_grid_ys[i]
            color = self._colors['full_cell']
            if i in self._free_spawn_cells:
                color = self._colors['free_cell']
            self._canvas.draw_dot(b2Vec2(x,y), depth=0, color=color)
        rendering.draw_world(self._canvas, self._world, self._colors, 
                             self._outline_colors)
        for i, agent in enumerate(self._agents):
            rendering.draw_lidar(
                self._canvas, n_lasers=self._n_lasers, 
                fov=self._lidar_angle, radius=self._lidar_depth, 
                transform=agent.transform, scan=self._obs[i], 
                on=self._colors['lidar_on'], off=self._colors['lidar_off'])
            hand = agent.userData.get('holds', None)
            hand_color = self._colors['hand_on'] if hand is not None \
                         else self._colors['hand_off']
            rendering.draw_laser(
              self._canvas, origin=agent.position, angle=agent.angle, 
              depth=self._hold_range, scan=None, on=hand_color, 
              off=hand_color)
        return self._canvas.render()

    def close(self) -> None:
        if self._canvas is not None:
            self._canvas.close()

