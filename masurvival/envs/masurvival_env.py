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
    Health, Heal, Melee, Item, item_prototype, Inventory, AutoPickup, UseLast, 
    DeathDrop, SafeZone, BattleRoyale, GiveLast, Object, ObjectItem,
    ImmunityPhase)

Vec = Tuple[float, ...]
# Self, lidars, other agents, inventory, items, objects
AgentObservation = \
    Tuple[Vec, List[float], List[Vec], List[Vec], List[Vec], List[Vec]]
Observation = Tuple[AgentObservation, ...]
AgentAction = Tuple[int, ...]
Action = Tuple[AgentAction, ...]
Reward = Vec

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
        'damage': 10,
        'drift': True, # so that we can actually render actions
    },
    'boxes': {
        'n_boxes': 2,
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

    # Whether agents can see everything or only things their "cameras" see.
    omniscent: bool

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
    steps: int
    spawner: SpawnGrid

    @property
    def n_agents(self):
        n = self.config['agents'].get('n_spawns', None)
        if n is None:
            return self.config['agents']['n_agents']
        return n

    def __init__(
            self, config: Optional[Config] = None, omniscent: bool = False):
        self.omniscent = omniscent
        self.config = recursive_apply(lambda _: _, self.config)
        if config is not None:
            for k, subconfig in config.items():
                self.config[k] |= subconfig
        self.observation_space, self.observation_sizes = self._compute_obs_space()
        n_agents = self.config['agents']['n_agents']
        actions_spaces = (self.agent_action_space,)*n_agents # type: ignore
        self.action_space = spaces.Tuple(actions_spaces)
        spawn_grid_config = self.config['spawn_grid']
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
        self.spawner = SpawnGrid(**spawn_grid_config)
        agents_modules = [
            ResetSpawns(spawner=self.spawner, **agents_config),
            #sim.LogDeaths(), # this must come before IndexBodies to work
            sim.IndexBodies(),
            sim.Cameras(**cameras_config),
            sim.Lidars(**lidar_config),
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
            BattleRoyale()]
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
            Heal(**heals_heal_config),]
            #Item(),]
        room_size = spawn_grid_config['floor_size']
        groups = {
            'boxes': sim.Group(boxes_modules),
            'box_items': sim.Group([boxes_items]),
            'heals': sim.Group(heals_modules),
            'walls': sim.Group([ThickRoomWalls(room_size)]),
            'agents': sim.Group(agents_modules)
        }
        self.simulation = sim.Simulation(groups=groups)

    def _compute_obs_space(self):
        n_lasers = self.config['lidars']['n_lasers']
        n_agents = self.n_agents
        n_boxes = self.config['boxes']['n_boxes']
        n_heals = self.config['heals']['reset_spawns']['n_items']
        n_walls = 4
        agent_size = 1+3+3
        item_size = 1+3
        object_size = 1+2+3
        inventory_item_size = 1
        inventory_slots = self.config['inventory']['slots']
        R = dict(low=float('-inf'), high=float('inf'))
        B = dict(low=0., high=1.)
        space = spaces.Tuple([
            spaces.Box(**R, shape=(n_agents, agent_size,)),
            spaces.Box(**R, shape=(n_agents, n_lasers,)),
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Box(**R, shape=(n_agents, n_agents-1, agent_size)),
                    spaces.Box(**B, shape=(n_agents, n_agents-1,)),
                ]),
                spaces.Tuple([
                    spaces.Box(**R, shape=(n_agents, inventory_slots, inventory_item_size)),
                    spaces.Box(**B, shape=(n_agents, inventory_slots,))
                ]),
                spaces.Tuple([
                    spaces.Box(**R, shape=(n_agents, n_heals+n_boxes, item_size)),
                    spaces.Box(**B, shape=(n_agents, n_heals+n_boxes,))
                ]),
                spaces.Tuple([
                    spaces.Box(**R, shape=(n_agents, n_walls+n_boxes, object_size)),
                    spaces.Box(**B, shape=(n_agents, n_walls+n_boxes,))
                ]),
            ])
        ])
        return space, [agent_size, n_lasers, [agent_size, inventory_item_size, item_size, object_size]]

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
        self.simulation.groups['agents'].get(DeathDrop)[0].rng = self.rng
        obs = self.fetch_observations(self.simulation.groups['agents'])
        info: Dict = {}
        self.steps = 0
        return (obs, info) if return_info else obs # type: ignore

    def step(self, # type: ignore
             action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        assert self.action_space.contains(action), "Invalid action."
        agents = self.simulation.groups['agents']
        self.queue_actions(agents, action)
        self.simulation.step()
        obs = self.fetch_observations(agents)
        reward = self.compute_rewards(agents, **self.config['reward_scheme'])
        done = agents.get(BattleRoyale)[0].over
        info: Dict = {}
        self.steps += 1
        return obs, reward, done, info # type: ignore

    # Ignores the render mode after the first call. TODO change that
    # Always returns the rendered frame, even with 'human' mode.
    def render(self, mode: str = 'human') -> np.ndarray:
        import masurvival.rendering as rendering
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render mode: {mode}')
        if self.canvas is None:
            views = {
                self.simulation.groups['agents']: [
                    rendering.SafeZone(
                        **rendering.safe_zone_view_config), # type: ignore
                    rendering.ImmunityCooldown(
                        **rendering.immunity_view_config), # type: ignore
                    rendering.Bodies(
                        **rendering.agent_bodies_view_config), # type: ignore
                    rendering.BodyIndices(
                        **rendering.body_indices_view_config), # type: ignore
                    #rendering.Lidars(
                    #    **rendering.agent_lidars_view_config), # type: ignore
                    rendering.Cameras(
                        **rendering.cameras_view_config), # type: ignore
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
        lidars = agents.get(sim.Lidars)[0]
        inventories = agents.get(Inventory)[0]
        # Reverse lists so we can pop from the end to get observations in 
        # order of agent body index.
        if not self.omniscent:
            seens = agents.get(sim.Cameras)[0].seen
        else:
            everything = []
            for g in self.simulation.groups.values():
                everything += g.bodies
            seens = [list(everything) for _ in range(self.n_agents)]
            for i, agent_body in enumerate(bodies):
                seens[i].remove(agent_body)
        seens = list(reversed(seens))
        scans = list(reversed(lidars.scans))
        obs_dead = lambda i: (
            (i,0,0,0,0,0,0), [lidars.depth]*lidars.n_lasers, [], [], [], [])
        obss = []
        for agent_id, body in enumerate(bodies):
            if body is None:
                obss.append(obs_dead(agent_id))
                continue
            self_: Vec = (
                agent_id, *body.position, body.angle, *body.linearVelocity, 
                body.angularVelocity)
            depths = [lidars.depth if scan is None else scan[1]
                      for scan in  scans.pop()]
            seen = seens.pop()
            inventory_items = inventories.inventories[body]
            inventory = []
            for item in inventory_items:
                type_ = 0 if isinstance(item, Heal) else 1
                item_obs: Vec = (float(type_),)
                inventory.append(item_obs)
            others = []
            items = []
            objects = []
            for b in seen:
                group = sim.Group.body_group(b)
                qpos: Vec = (*b.position, b.angle)
                # Object types: 0 boxes, 1 walls
                # Item types: 0 heals, 1 broken boxes
                if group is self.simulation.groups['agents']:
                    other_agent_id = [c == b for i, c in enumerate(bodies)][0]
                    qvel: Vec = (*b.linearVelocity, b.angularVelocity)
                    others.append((other_agent_id, *qpos, *qvel))
                elif group is self.simulation.groups['boxes']:
                    type_ = 0
                    assert isinstance(b.fixtures[0].shape, b2PolygonShape)
                    try:
                        size = sim.rect_dimensions(b.fixtures[0].shape)
                    except AssertionError:
                        print(b.fixtures[0].shape.vertices)
                        raise
                    objects.append((*qpos, float(type_), *size))
                elif group is self.simulation.groups['box_items']:
                    type_ = 1
                    items.append((*qpos, float(type_)))
                elif group is self.simulation.groups['heals']:
                    type_ = 0
                    items.append((*qpos, float(type_)))
                elif group is self.simulation.groups['walls']:
                    type_ = 1
                    size = sim.rect_dimensions(b.fixtures[0].shape)
                    objects.append((*qpos, float(type_), *size))
                else:
                    assert False, 'How did we get here?'
            obss.append((self_, depths, others, inventory, items, objects))

        obss_np = _zero_element(self.observation_space)
        for i in [0,1]:
            obss_np[i] = np.array([obs[i] for obs in obss], dtype=np.float32)
        for i in range(4):
            for j, obs in enumerate(obss):
                if len(obs[i+2]) > 0:
                    obss_np[2][i][0][j,:len(obs[i+2])] = np.array(obs[i+2], dtype=np.float32)
                    obss_np[2][i][1][j,:len(obs[i+2])] = 1
        #assert self.observation_space.contains(obss_np)
        return obss_np

    def queue_actions(self, agents: sim.Group, all_actions: Action):
        d = [0., 1., -1.]
        bodies = agents.get(sim.IndexBodies)[0].bodies
        actions = []
        for i, a in enumerate(all_actions):
            if bodies[i] is not None:
                actions.append(a)
        motor_controls = [(d[a[0]], d[a[1]], d[a[2]]) for a in actions]
        agents.get(sim.DynamicMotors)[0].controls = motor_controls
        agents.get(Melee)[0].attacks = [bool(a[3]) for a in actions]
        agents.get(UseLast)[0].uses = [bool(a[4]) for a in actions]
        agents.get(GiveLast)[0].give = [bool(a[5]) for a in actions]

    def compute_rewards(
            self, agents: sim.Group, sparse: bool) -> Tuple[float, ...]:
        bodies = agents.get(sim.IndexBodies)[0].bodies
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        if not sparse:
            rewards += np.array([-1 if body is None else +1 for body in bodies])
        game = agents.get(BattleRoyale)[0]
        if game.over:
            agent_health = self.config['health']['health']
            n_heals = self.config['heals']['reset_spawns']['n_spawns']
            healing = self.config['heals']['heal']['healing']
            max_prize = agents.get(SafeZone)[0].max_lifespan(agent_health + n_heals*healing)
            prize = max_prize - self.steps
            rewards += np.array([prize if won else -prize for won in game.results])
        return rewards


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

