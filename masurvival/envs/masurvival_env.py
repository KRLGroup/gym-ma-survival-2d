from typing import Any, Optional, Union, Tuple, List, Dict, Set, Callable
from types import ModuleType
from operator import attrgetter
import math

import numpy as np
import gym
from gym import spaces

from Box2D import ( # type: ignore
  b2World, b2Body, b2Fixture, b2Joint, b2Vec2, b2PolygonShape)

import masurvival.simulation as sim
from masurvival.semantics import (
    SpawnGrid, ResetSpawns, agent_prototype, box_prototype, ThickRoomWalls, 
    Health, Heal, ContinuousMelee, Item, item_prototype, Inventory, 
    AutoPickup, UseLast, DeathDrop, SafeZone, BattleRoyale, GiveLast, Object, 
    ObjectItem, ImmunityPhase, Melee)

Vec = Tuple[float, ...]
# Self, lidars, other agents, inventory, items, objects
AgentObservation = \
    Tuple[Vec, List[float], List[Vec], List[Vec], List[Vec], List[Vec]]
Observation = Tuple[AgentObservation, ...]
AgentAction = Tuple[int, ...]
Action = Tuple[AgentAction, ...]
Reward = Vec

Config = Dict[str, Dict[str, Any]]

class BaseEnv(gym.Env):

    # See the default values for available params for each subclass.
    # It is expected that each subclass defines this at class level.
    config: Dict[str, Any] = NotImplemented
    # actions and observation spaces
    action_space: spaces.Space # changes based on number of agents
    observation_space: spaces.Space = NotImplemented
    # rendering
    metadata: Dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    window_size: int = 512 # only affects the first render call
    canvas: Optional[Any] = None
    # internal state
    simulation: sim.Simulation
    steps: int

    @property
    def n_agents(self):
        n = self.config['agents'].get('n_spawns', None)
        if n is None:
            return self.config['agents']['n_agents']
        return n

    def __init__(self, config: Optional[Config] = None):
        self.np_random = np.random.default_rng()
        self.config = recursive_apply(lambda _: _, self.config)
        if config is not None:
            for k, subconfig in config.items():
                self.config[k] |= subconfig
        self.observation_space = self.compute_obs_space()
        self.action_space = self.compute_action_space()

    #TODO fix seed behaviour; for now it uses old interface so internal seed is not reset on reset, but only on init
    def reset(self, seed: int = None, return_info: bool = False,
              options: Optional[Dict[Any, Any]] = None) \
            -> Union[Observation,
                     Tuple[Observation, Dict[Any, Any]]]:
        #if seed is None and 'seed' in self.config['rng']:
        #    seed = self.config['rng']['seed']
        #super().reset(seed=seed)
        #super().reset()
        self.spawner.rng = self.np_random
        self.spawner.reset()
        self.pre_reset()
        self.simulation.reset()
        obs = self.fetch_observations(self.simulation.groups['agents'])
        info: Dict = {}
        self.steps = 0
        return (obs, info) if return_info else obs # type: ignore

    def step(self, # type: ignore
             actions: Action) -> Tuple[Observation, Reward, bool, Dict]:
        actions = tuple(a for a in actions)
        assert self.action_space.contains(actions), f"Invalid action {actions}."
        agents = self.simulation.groups['agents']
        self.queue_actions(agents, actions)
        self.simulation.step()
        obs = self.fetch_observations(agents)
        rewards = self.compute_rewards(agents, **self.config['reward_scheme'])
        done = self.is_done(agents)
        info: Dict = {}
        self.steps += 1
        return obs, rewards, done, info # type: ignore

    # Ignores the render mode after the first call. TODO change that
    # Always returns the rendered frame, even with 'human' mode.
    def render(self, mode: str = 'human') -> np.ndarray:
        import masurvival.rendering as rendering
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if self.canvas is None:
            views = self.init_views()
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

    def pre_reset(self):
        raise NotImplementedError

    def fetch_observations(self, agents: sim.Group) -> Observation:
        raise NotImplementedError

    def queue_actions(self, agents: sim.Group, all_actions: Action):
        raise NotImplementedError

    def compute_rewards(
            self, agents: sim.Group, sparse: bool) -> Tuple[float, ...]:
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def init_views(self):
        raise NotImplementedError


# 1v1 variant, heals only

onevsone_heals_default_config: Config = {
    'observation': {
        'omniscent': True,
    },
    'reward_scheme': {
        'r_alive': 1,
        'r_dead': -1,
    },
    'gameover' : {
        'mode': 'alldead', # in {alldead, lastalive}
    },
    'rng': { 'seed': 42 },
    'spawn_grid': {
        'grid_size': 4,
        'floor_size': 20,
    },
    # disabled in the actual env
    'immunity_phase': {
        'cooldown': 300,
    },
    'agents': {
        'n_agents': 2,
        'agent_size': 1,
    },
    'cameras': {
        'fov': 0.4*np.pi,
        'depth': 10,
    },
    'lidars': {
        'n_lasers': 8,
        'fov': 0.8*np.pi,
        'depth': 2,
    },
    'motors': {
        'impulse': (0.25, 0.25, 0.0125),
        'drift': False,
    },
    'health': {
        'health': 100,
    },
    'melee': {
        'range': 2,
        'damage': 20,
        'cooldown': 40, # makes sense as long as its greater than damage
        'drift': True, # so that we can actually render actions
    },
    'boxes': {
        'n_boxes': 0,
        'box_size': 1,
    },
    'boxes_item': {
        'item_size': 0.5,
        'offset': 0.75,
    },
    'boxes_health': {
        'health': 20,
    },
    'heals': {
        'reset_spawns': {
            'n_items': 4,
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
        'centers': 'random',
    },
}

class OneVsOneHeals(BaseEnv):

    config = onevsone_heals_default_config

    def __init__(self, config=None):
        super().__init__(config)
        spawn_grid_config = self.config['spawn_grid']
        room_size = spawn_grid_config['floor_size']
        agents_config = self.config['agents']
        agent_size = agents_config.pop('agent_size')
        agents_config['prototype'] = agent_prototype(agent_size)
        agents_config['n_spawns'] = agents_config.pop('n_agents')
        immunity_phase_config = self.config['immunity_phase']
        cameras_config = self.config['cameras']
        lidar_config = self.config['lidars']
        motor_config = self.config['motors']
        health_config = self.config['health']
        melee_config = self.config['melee']
        inventory_config = self.config['inventory']
        auto_pickup_config = self.config['auto_pickup']
        give_config = self.config['give']
        death_drop_config = self.config['death_drop']
        safe_zone_config = self.config['safe_zone']
        safe_zone_config['room_size'] = room_size
        self.spawner = SpawnGrid(**spawn_grid_config)
        agents_modules = [
            ResetSpawns(spawner=self.spawner, **agents_config),
            #sim.LogDeaths(), # this must come before IndexBodies to work
            sim.IndexBodies(),
            sim.Cameras(**cameras_config),
            #sim.Lidars(**lidar_config),
            sim.DynamicMotors(**motor_config),
            DeathDrop(**death_drop_config), # needs to be before inventory
            Inventory(**inventory_config),
            Health(**health_config),
            AutoPickup(**auto_pickup_config),
            UseLast(),
            GiveLast(**give_config),
            SafeZone(**safe_zone_config),
            #ImmunityPhase(**immunity_phase_config),
            Melee(**melee_config), # needs to be after anything that kills bodies
            #BattleRoyale(),
            ]
        boxes_config = self.config['boxes']
        boxes_item_config = self.config['boxes_item']
        boxes_health_config = self.config['boxes_health']
        box_size = boxes_config.pop('box_size')
        boxes_config['prototype'] = box_prototype(box_size)
        boxes_config['n_spawns'] = boxes_config.pop('n_boxes')
        boxes_item_size = boxes_item_config.pop('item_size')
        boxes_item_config['prototype'] = item_prototype(boxes_item_size)
        boxes_items = ObjectItem(**boxes_item_config)
        boxes_items_modules = [boxes_items]
        boxes_modules = [
            ResetSpawns(spawner=self.spawner, **boxes_config),
            Object(boxes_items),
            Health(**boxes_health_config),]
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
            sim.IndexBodies(),
            Heal(**heals_heal_config),]
            #Item(),]
        groups = {
            'boxes': sim.Group(boxes_modules),
            'box_items': sim.Group([boxes_items]),
            'heals': sim.Group(heals_modules),
            'walls': sim.Group([ThickRoomWalls(room_size)]),
            'agents': sim.Group(agents_modules)
        }
        self.simulation = sim.Simulation(groups=groups)

    def compute_obs_space(self):
        n_lasers = self.config['lidars']['n_lasers']
        n_heals = self.config['heals']['reset_spawns']['n_items']
        agent_size = 1+1+3+3 # ID, health, qpos, qvel
        safe_zone_size = 3+3 # center & radius + next center & radius
        item_size = 2 # pos
        R = dict(low=float('-inf'), high=float('inf'))
        mask_spaces = {}
        if not self.config['observation']['omniscent']:
            # The masks have 0 for visible entities and 1 otherwise. This is to allow easy conversion to the pytorch multihead attention masks, which require True for unattendable positions.
            mask_spaces = {
                'heals_mask': spaces.Box(**R, shape=(
                    self.n_agents, n_heals)
                ),
                'zone_mask': spaces.Box(**R, shape=(self.n_agents, 1)),
            }
        zone_shape = \
            (self.n_agents, safe_zone_size) \
            if self.config['observation']['omniscent'] \
            else (self.n_agents, 1, safe_zone_size)
        space = spaces.Dict({
            'agent': spaces.Box(**R, shape=(self.n_agents, agent_size)),
            'other': spaces.Box(**R, shape=(self.n_agents, agent_size)),
            'zone': spaces.Box(**R, shape=zone_shape),
            'heals': spaces.Box(**R, shape=(
                self.n_agents, n_heals, item_size)
            ),
            **mask_spaces,
        })
        return space

    def compute_action_space(self):        
        n_agents = self.config['agents']['n_agents']
        single_action_space = spaces.MultiDiscrete([3,3,3,2,2,2])
        actions_spaces = (single_action_space,)*n_agents # type: ignore
        return spaces.Tuple(actions_spaces)

    def pre_reset(self):
        self.simulation.groups['agents'].get(SafeZone)[0].rng = self.np_random
        self.simulation.groups['agents'].get(DeathDrop)[0].rng = self.np_random

    def fetch_observations(self, agents: sim.Group) -> Observation:
        agent_bodies = agents.get(sim.IndexBodies)[0].bodies
        health = agents.get(Health)[0]
        safe_zone = agents.get(SafeZone)[0]
        x = {}
        x['agent'] = self._fetch_self_observations(agent_bodies, health)
        next_zone = None
        if safe_zone.phase < safe_zone.phases-1:
            next_zone = [
                *safe_zone.centers[safe_zone.phase+1],
                safe_zone.radiuses[safe_zone.phase+1],
            ]
        else:
            next_zone = [0, 0, 0]
        x_zone = np_floats([
            *safe_zone.zone[1].position,
            safe_zone.zone[0].radius,
            *next_zone,
        ])
        assert self.n_agents == 2
        x['other'] = np.vstack([x['agent'][1], x['agent'][0]])
        x['zone'] = np.tile(x_zone, [self.n_agents, 1])
        if not self.config['observation']['omniscent']:
            x['zone'] = np.expand_dims(x['zone'], axis=1)
            x['zone_mask'] = np_float_zeros(
                self.observation_space['zone_mask'].shape
            )
        x_heals = np_float_zeros(
            self.observation_space['heals'].shape[1:]
        )
        for i, heal in enumerate(self.simulation.groups['heals'].bodies):
            x_heals[i] = np_floats([*heal.position])
        x['heals'] = np.tile(x_heals, [self.n_agents, 1, 1])
        if not self.config['observation']['omniscent']:
            x['heals_mask'] = self._fetch_heals_mask(
                agents, agent_bodies
            )
        assert self.observation_space.contains(x), f'{x} not contained in the observation space {self.observation_space}'
        return x

    def _fetch_self_observations(self, agent_bodies, health):
        x = np_float_zeros(self.observation_space['agent'].shape)
        for i, agent_body in enumerate(agent_bodies):
            if agent_body is None:
                #TODO does it make sense to give the agent a very far away position?
                x[i][0] = i
                continue
            x[i] = np_floats([
                i,
                health.healths[agent_body],
                *agent_body.position,
                agent_body.angle,
                *agent_body.linearVelocity,
                agent_body.angularVelocity
            ])
        return x

    def _fetch_heals_mask(self, agents: sim.Group, agent_bodies):
        masks = np_float_ones(self.observation_space['heals_mask'].shape)
        cameras = agents.get(sim.Cameras)[0]
        heals = self.simulation.groups['heals'].get(sim.IndexBodies)[0].bodies
        for a_body, seen in zip(agents.bodies, cameras.seen):
            agent_id = agent_bodies.index(a_body)
            masks[agent_id] = np_floats([
                0 if h in seen else 1 for h in heals
            ])
        return masks

    def queue_actions(self, agents: sim.Group, all_actions: Action):
        d = [-1., 0., 1.]
        bodies = agents.get(sim.IndexBodies)[0].bodies
        actions = []
        #TODO check actions are applied to the correct bodies
        for i, a in enumerate(all_actions):
            if bodies[i] is not None:
                actions.append(a)
        motor_controls = [(d[a[0]], d[a[1]], d[a[2]]) for a in actions]
        agents.get(sim.DynamicMotors)[0].controls = motor_controls
        agents.get(Melee)[0].attacks = [bool(a[3]) for a in actions]
        agents.get(UseLast)[0].uses = [bool(a[4]) for a in actions]
        agents.get(GiveLast)[0].give = [bool(a[5]) for a in actions]
        #agents.get(UseLast)[0].uses = [bool(a[3]) for a in actions]
        #agents.get(GiveLast)[0].give = [bool(a[4]) for a in actions]

    def compute_rewards(
            self, agents: sim.Group,
            r_alive: float,
            r_dead: float
    ) -> Tuple[float, ...]:
        bodies = agents.get(sim.IndexBodies)[0].bodies
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        rewards += np.array([
            r_dead if body is None else r_alive for body in bodies
        ])
#         game = agents.get(BattleRoyale)[0]
#         if game.over:
#             agent_health = self.config['health']['health']
#             n_heals = self.config['heals']['reset_spawns']['n_spawns']
#             healing = self.config['heals']['heal']['healing']
#             max_prize = agents.get(SafeZone)[0].max_lifespan(agent_health + n_heals*healing)
#             prize = 1. if sparse else (max_prize - self.steps) / max_prize
#             rewards += np.array([prize if won else -prize for won in game.results])
        return rewards

    def is_done(self, agents):
        #done = agents.get(BattleRoyale)[0].over
        done = False
        n_alive = len([
            b for b in agents.get(sim.IndexBodies)[0].bodies
            if b is not None
        ])
        if self.config['gameover']['mode'] == 'alldead':
            done = n_alive == 0
        elif self.config['gameover']['mode'] == 'lastalive':
            done = n_alive == 1
        else:
            assert False, 'Invalid gameover mode'
        #agent_health = self.config['health']['health']
        #n_heals = self.config['heals']['reset_spawns']['n_spawns']
        #healing = self.config['heals']['heal']['healing']
        #max_steps = agents.get(SafeZone)[0].max_lifespan(agent_health + n_heals*healing)
        #print(f'max_steps = {max_steps}')
        #done = self.steps >= max_steps
        if done:
            print(f'done in {self.steps} steps')
        return done

    def init_views(self):
        import masurvival.rendering as rendering
        views = {
            self.simulation.groups['agents']: [
                rendering.SafeZone(
                    **rendering.safe_zone_view_config), # type: ignore
                #rendering.ImmunityCooldown(
                #    **rendering.immunity_view_config), # type: ignore
                rendering.Bodies(
                    **rendering.agent_bodies_view_config), # type: ignore
                rendering.BodyIndices(
                    **rendering.body_indices_view_config), # type: ignore
                #rendering.Lidars(
                #    **rendering.agent_lidars_view_config), # type: ignore
                #rendering.Cameras(
                #    **rendering.cameras_view_config), # type: ignore
                rendering.Health(
                    **rendering.health_view_config), # type: ignore
                rendering.Melee(
                    **rendering.melee_view_config), # type: ignore
                #rendering.Inventory(
                #    **rendering.inventory_view_config), # type: ignore
                #rendering.GiveLast(
                #    **rendering.give_view_config), # type: ignore
            ],
            self.simulation.groups['boxes']: [
                rendering.Bodies(
                    **rendering.bodies_view_config), # type: ignore
            ],
            self.simulation.groups['box_items']: [
                rendering.Bodies(
                    **rendering.bodies_view_config), # type: ignore
            ],
            self.simulation.groups['heals']: [
                rendering.Bodies(
                    **rendering.bodies_view_config), # type: ignore
            ],
            self.simulation.groups['walls']: [
                rendering.Bodies(
                    **rendering.walls_view_config), # type: ignore
            ],
        }
        return views





# only supports tuple and box spaces for now; returns lists instead of tuples to support mutability
def _zero_element(space: gym.spaces.Space, dtype=np.float32):
    if isinstance(space, spaces.Tuple):
        return [_zero_element(subspace) for subspace in space]
    elif isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=dtype)
    else:
        assert False, f'Unsupported space of type {type(space)}'


#TODO import this from the policy opt utils
# The "structure" of the arguments is taken from the 'struct_arg', which is the last by default.
def recursive_apply(f: Callable, *args: Any, default: Optional[Any] = None, struct_arg: int = -1) -> Any:
    if len(args) == 0:
        return default
    if isinstance(args[struct_arg], list):
        return [recursive_apply(f, *arg) for arg in zip(*args)]
    if isinstance(args[struct_arg], dict):
        return {k: recursive_apply(f, *[arg[k] for arg in args])
                for k in args[struct_arg].keys()}
    else:
        return f(*args)


def np_floats(a):
    return np.array(a, dtype=np.float32)

def np_float_zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def np_float_ones(shape):
    return np.ones(shape, dtype=np.float32)

