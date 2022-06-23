from typing import Any, Optional, Union, Tuple, List, Dict, Set, Callable
from types import ModuleType
from operator import attrgetter
import math

import numpy as np
import gym
from gym import spaces

from Box2D import b2World, b2Body, b2Fixture, b2Joint, b2Vec2 # type: ignore

import masurvival.simulation as sim
from masurvival.semantics import (
    SpawnGrid, ResetSpawns, agent_prototype, box_prototype, ThickRoomWalls, 
    Health, Heal, Melee, Item, item_prototype, Inventory, AutoPickup, UseLast, 
    DeathDrop, SafeZone, BattleRoyale, GiveLast)


AgentObservation = List[sim.LaserScan]
Observation = Tuple[AgentObservation, ...]
AgentAction = Tuple[int, int, int, int, int, int]
Action = Tuple[AgentAction, ...]
Reward = Tuple[float, ...]

Config = Dict[str, Dict[str, Any]]

default_config: Config = {
    'reward_scheme': {
        # if sparse, reward only at episode end (+1 win, -1 lose)
        # if not sparse, -1 reward if dead, +1 if alive
        'sparse': False,
    },
    'rng': { 'seed': 42 },
    'spawn_grid': {
        'grid_size': 4,
        'floor_size': 20,
    },
    'agents': {
        'n_agents': 4,
        'agent_size': 1,
    },
    'lidars': {
        'n_lasers': 8,
        'fov': 0.8*np.pi,
        'depth': 2,
    },
    'motors': {
        'impulse': (0.5, 0.5, 0.025),
        'drift': False,
    },
    'health': {
        'health': 100,
    },
    'melee': {
        'range': 2,
        'damage': 10,
        'drift': True, # so that we can actually render actions
    },
    'boxes': {
        'n_boxes': 2,
        'box_size': 1,
    },
    'heals': {
        'reset_spawns': {
            'n_items': 3,
            'item_size': 0.5,
        },
        'heal': {
            'healing': 50,
        },
    },
    'inventory': {
        'slots': 4,
    },
    'auto_pickup': {
        'shape': sim.circle_shape(0.5),
    },
    'give': {
        'shape': sim.circle_shape(2),
    },
    'death_drop': {
        'radius': 0.5,
    },
    'safe_zone': {
        'phases': 5,
        'cooldown': 100,
        'damage': 1,
        'radiuses': [10, 5, 2.5, 1],
        'centers': [(0,0), (0,0), (0,0), (0,0)],
    },
}

class MaSurvivalEnv(gym.Env):

    # See the default values for available params
    config: Dict[str, Any] = default_config
    # actions and observation spaces
    agent_action_space: spaces.Space = spaces.MultiDiscrete([3,3,3,2,2,2])
    action_space: spaces.Space # changes based on number of agents
    observation_space: spaces.Space = NotImplemented
    # rendering
    metadata: Dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    window_size: int = 512 # only affects the first render call
    canvas: Optional[Any] = None
    # internal state
    simulation: sim.Simulation
    spawner: SpawnGrid

    def __init__(self, config: Optional[Config] = None):
        if config is not None:
            for k, subconfig in config.items():
                config[k] |= subconfig
        n_agents = self.config['agents']['n_agents']
        actions_spaces = (self.agent_action_space,)*n_agents # type: ignore
        self.action_space = spaces.Tuple(actions_spaces)
        spawn_grid_config = self.config['spawn_grid']
        agents_config = self.config['agents']
        agent_size = agents_config.pop('agent_size')
        agents_config['prototype'] = agent_prototype(agent_size)
        agents_config['n_spawns'] = agents_config.pop('n_agents')
        lidar_config = self.config['lidars']
        motor_config = self.config['motors']
        health_config = self.config['health']
        melee_config = self.config['melee']
        inventory_config = self.config['inventory']
        auto_pickup_config = self.config['auto_pickup']
        give_config = self.config['give']
        death_drop_config = self.config['death_drop']
        safe_zone_config = self.config['safe_zone']
        self.spawner = SpawnGrid(**spawn_grid_config)
        agents_modules = [
            ResetSpawns(spawner=self.spawner, **agents_config),
            sim.LogDeaths(), # this must come before IndexBodies to work
            sim.IndexBodies(),
            sim.Lidars(**lidar_config),
            sim.DynamicMotors(**motor_config),
            Health(**health_config),
            Melee(**melee_config),
            DeathDrop(**death_drop_config), # needs to be before inventory
            Inventory(**inventory_config),
            AutoPickup(**auto_pickup_config),
            UseLast(),
            GiveLast(**give_config),
            #SafeZone(**safe_zone_config),
            BattleRoyale()]
        boxes_config = self.config['boxes']
        box_size = boxes_config.pop('box_size')
        boxes_config['prototype'] = box_prototype(box_size)
        boxes_config['n_spawns'] = boxes_config.pop('n_boxes')
        boxes_modules = [
            ResetSpawns(spawner=self.spawner, **boxes_config),]
        heals_config = self.config['heals']
        heals_spawn_config = heals_config['reset_spawns']
        heals_heal_config = heals_config['heal']
        heal_size = heals_spawn_config.pop('item_size')
        heal_prototype = item_prototype(heal_size)
        heals_heal_config['prototype'] = heal_prototype
        heals_spawn_config['prototype'] = heal_prototype
        heals_spawn_config['n_spawns'] = heals_spawn_config.pop('n_items')
        heals_modules = [
            ResetSpawns(spawner=self.spawner, **heals_spawn_config),
            Heal(**heals_heal_config),]
            #Item(),]
        agents = sim.Group(agents_modules)
        boxes = sim.Group(boxes_modules)
        heals = sim.Group(heals_modules)
        room_size = spawn_grid_config['floor_size']
        walls = sim.Group([ThickRoomWalls(room_size)])
        self.simulation = sim.Simulation(groups=[agents, boxes, heals, walls])

    # if given, the seed takes precedence over the config seed
    def reset(self, seed: int = None, return_info: bool = False,
              options: Optional[Dict[Any, Any]] = None) \
            -> Union[Observation,
                     Tuple[Observation, Dict[Any, Any]]]:
        if seed is None and 'seed' in self.config['rng']:
            seed = self.config['rng']['seed']
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        self.spawner.rng = self.rng
        self.spawner.reset()
        self.simulation.reset()
        self.simulation.groups[0].get(DeathDrop)[0].rng = self.rng
        obs = self.fetch_observations(self.simulation.groups[0])
        info: Dict = {}
        return (obs, info) if return_info else obs # type: ignore

    def step(self, # type: ignore
            action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        assert self.action_space.contains(action), "Invalid action."
        agents, boxes, heals, walls = self.simulation.groups
        self.queue_actions(agents, action)
        self.simulation.step()
        obs = self.fetch_observations(agents)
        reward = self.compute_rewards(agents, **self.config['reward_scheme'])
        done = agents.get(BattleRoyale)[0].over
        if done:
            print(reward)
        info: Dict = {}
        return obs, reward, done, info # type: ignore

    # Ignores the render mode after the first call. TODO change that
    # Always returns the rendered frame, even with 'human' mode.
    def render(self, mode: str = 'human') -> np.ndarray:
        import masurvival.rendering as rendering
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if self.canvas is None:
            agents, boxes, heals, walls = self.simulation.groups
            views = {
                agents: [
                    rendering.SafeZone(
                        **rendering.safe_zone_view_config), # type: ignore
                    rendering.Bodies(
                        **rendering.agent_bodies_view_config), # type: ignore
                    rendering.BodyIndices(
                        **rendering.body_indices_view_config), # type: ignore
                    rendering.Lidars(
                        **rendering.agent_lidars_view_config), # type: ignore 
                    rendering.Health(
                        **rendering.health_view_config), # type: ignore
                    rendering.Melee(
                        **rendering.melee_view_config), # type: ignore
                    rendering.Inventory(
                        **rendering.inventory_view_config), # type: ignore
                    rendering.GiveLast(
                        **rendering.give_view_config), # type: ignore
                ],
                boxes: [
                    rendering.Bodies(
                        **rendering.bodies_view_config), # type: ignore
                ],
                heals: [
                    rendering.Bodies(
                        **rendering.bodies_view_config), # type: ignore
                ],
                walls: [
                    rendering.Bodies(
                        **rendering.walls_view_config), # type: ignore
                ],
            }
            self.canvas = rendering.Canvas(
                width=self.window_size, height=self.window_size, 
                world_size=self.config['spawn_grid']['floor_size'], 
                render_mode=mode, fps=self.metadata['render_fps'],
                background=rendering.background, views=views) # type: ignore
        self.canvas.clear()
        return self.canvas.render()

    def close(self) -> None:
        if self.canvas is not None:
            self.canvas.close()

    def fetch_observations(self, agents: sim.Group) -> Observation:
        bodies = agents.get(sim.IndexBodies)[0].bodies
        # Reverse scans so we can pop from the end of the list.
        scans = list(reversed(agents.get(sim.Lidars)[0].scans))
        obs = [[] if body is None else scans.pop() for body in bodies]
        return tuple(obs)

    def queue_actions(self, agents: sim.Group, actions: Action):
        d = [0., 1., -1.]
        bodies = agents.get(sim.IndexBodies)[0].bodies
        actions_alive = []
        for i, a in enumerate(actions):
            if bodies[i] is not None:
                actions_alive.append(a)
        actions = actions_alive
        motor_controls = [(d[a[0]], d[a[1]], d[a[2]]) for a in actions]
        agents.get(sim.DynamicMotors)[0].controls = motor_controls
        agents.get(Melee)[0].attacks = [bool(a[3]) for a in actions]
        agents.get(UseLast)[0].uses = [bool(a[4]) for a in actions]
        agents.get(GiveLast)[0].give = [bool(a[5]) for a in actions]

    def compute_rewards(
            self, agents: sim.Group, sparse: bool) -> Tuple[float, ...]:
        bodies = agents.get(sim.IndexBodies)[0].bodies
        if not sparse:
            return tuple(-1 if body is None else +1 for body in bodies)
        game = agents.get(BattleRoyale)[0]
        if not game.over:
            return tuple(0 for _ in bodies)
        return tuple(+1 if won else -1 for won in game.results)
