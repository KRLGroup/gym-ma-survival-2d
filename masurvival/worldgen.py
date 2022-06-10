from typing import Any, Union, Optional, Tuple, List, Dict, Set
import math

import numpy as np

from Box2D import ( # type: ignore
     b2World, b2Body, b2FixtureDef, b2PolygonShape, b2CircleShape,
     b2_dynamicBody, b2_staticBody, b2Vec2, b2Shape)

import masurvival.geometry as geo
#TODO decouple this module from the simulation module (just avoid adding bodies directly to the world)
import masurvival.simulation as simulation

def square_shape(side: float) -> b2Shape:
    return b2PolygonShape(box=(side/2., side/2.))

def rect_shape(width: float, height: float) -> b2Shape:
    return b2PolygonShape(box=(width/2., height/2.))

def circle_shape(radius: float) -> b2Shape:
    return b2CircleShape(radius=radius)

_density = 1.
_restitution = 0.
_damping = 0.8

BodyConf = Dict[str, Any]

# in these, relative_size is in [0.,1.] w.r.t. given world_size

# relative size controls diameter
def agent_body_conf(world_size: float, relative_size: float = 0.05,
                    tag: str = 'agent') -> BodyConf:
    return {
        'shape': circle_shape(radius=(world_size*relative_size)/2.),
        'dynamic': True,
        'density': 2.*_density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': {'tag': tag},}

def ramp_body_conf(world_size: float, relative_size: float = 0.05,
                   tag: str = 'ramp') -> BodyConf:
    return {
        'shape': square_shape(side=world_size*relative_size),
        'dynamic': True,
        'sensor': False,
        'ramp_edge': 0,
        'density': _density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': {'tag': tag, 'lockable': True, 'holdable': True},}

def box_body_conf(world_size: float, relative_size: float = 0.05,
                  movable: bool = True, tag: str = 'box') -> BodyConf:
    return {
        'shape': square_shape(side=world_size*relative_size),
        'dynamic': movable,
        'density': _density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': {'tag': tag, 'lockable': movable, 'holdable': movable},}

# horizontal wall; aspect_ratio is wall length / wall width
#TODO use box2d chain shapes for world border
def wall_body_conf(world_size: float, aspect_ratio=100.,
                   relative_size: float = 1., tag: str = 'wall') -> BodyConf:
    width = world_size*relative_size
    height = width/aspect_ratio
    return {
        'shape': rect_shape(width=width, height=height),
        'extruded': True,
        'dynamic': False,
        'density': _density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': {'tag': tag},}

def uniform_grid(cells_per_side: int, grid_size: float):
    centers = np.arange(cells_per_side)/cells_per_side \
                         + 0.5/cells_per_side
    centers = grid_size*centers - grid_size/2.
    xs, ys = np.meshgrid(centers, centers)
    xs, ys = xs.flatten(), ys.flatten()
    return xs, ys

def populate_world(world: b2World, world_size: float,
                   spawn_grid_xs: np.ndarray, spawn_grid_ys: np.ndarray, 
                   n_agents: int = 2, n_ramps: int = 2, n_boxes: int = 2, 
                   n_pillars: int = 2,
                   rng: Optional[np.random.Generator] = None) \
        -> Tuple[Dict[str, List[b2Body]], Set[int]]:

    rng = rng or np.random.default_rng()
    n_bodies = n_agents + n_ramps + n_pillars + n_boxes

    if len(spawn_grid_xs.shape) != 1:
        raise ValueError('spawn_grid_xs')
    if len(spawn_grid_ys.shape) != 1:
        raise ValueError('spawn_grid_ys')
    if spawn_grid_xs.shape[0] != spawn_grid_ys.shape[0]:
        raise ValueError('spawn_grid_xs, spawn_grid_ys')
    if spawn_grid_xs.shape[0] < n_bodies:
        raise ValueError(f'populate_world: too few spawning cells: '
                         f'{spawn_grid_xs.shape[0]}, expected at least '
                         f'{n_bodies}')
    
    room_rel_size = 0.95
    wall_aspect_ratio = 100.
    
    # body configurations
    agent_conf = agent_body_conf(world_size)
    ramp_conf = ramp_body_conf(world_size)
    box_conf = box_body_conf(world_size)
    pillar_conf = box_body_conf(world_size, movable=False, 
                                relative_size=0.02, tag='pillar')
    wall_conf = wall_body_conf(world_size, relative_size=room_rel_size, 
                               aspect_ratio=wall_aspect_ratio)

    # Place the world border walls.
    walls = []
    walls_offset = room_rel_size*world_size/2. \
                   - (world_size/wall_aspect_ratio)/2
    wall_placements = [
        ((0., walls_offset), 0.), # north wall
        ((0., -walls_offset), 0.), # south wall
        ((walls_offset, 0.), math.pi/2.), # east wall
        ((-walls_offset, 0.), math.pi/2.),] # west wall
    wall_beg_contact = lambda a,b: print('Begun wall contact')
    wall_end_contact = lambda a,b: print('Ended wall contact')
    for position, angle in wall_placements:
        wall = simulation.add_body(
            world, **wall_conf, position=position, angle=angle)
        wall.userData['begin_contact'] = wall_beg_contact
        wall.userData['end_contact'] = wall_end_contact
        walls.append(wall)


    # Configure random spawning: the world is divided into a grid of cells, 
    # and only one entity will be able to spawn in each cell.
    free_cells = set(range(spawn_grid_xs.shape[0]))
    spawning_conf = {
      'xs': spawn_grid_xs, 'ys': spawn_grid_ys, 'world': world, 'rng': rng,}
    
    # Randomly spawn entities in the configured spawning grid.
    agents, free_cells = random_spawns(n_agents, free_cells, agent_conf, 
                                       **spawning_conf)
    ramps, free_cells = random_spawns(n_ramps, free_cells, ramp_conf, 
                                       **spawning_conf)
    boxes, free_cells = random_spawns(n_boxes, free_cells, box_conf,
                                      **spawning_conf)
    pillars, free_cells = random_spawns(n_pillars, free_cells, pillar_conf,
                                       **spawning_conf)
    bodies = {
      'agents': agents,
      'ramps': ramps,
      'boxes': boxes,
      'pillars': pillars,
      'walls': walls,}

    return bodies, free_cells


# generates random positions among the given ones
def random_spawns(n_spawns: int, free_positions: Set[int],
                  body_conf: BodyConf, xs: np.ndarray, ys: np.ndarray, 
                  world: b2World, rng: np.random.Generator) \
        -> Tuple[List[b2Body], Set[int]]:
    spawn_position_ids = rng.choice(list(free_positions), size=n_spawns, 
                                    replace=False)
    bodies = []
    for position_id in spawn_position_ids:
        position = xs[position_id], ys[position_id]
        body = simulation.add_body(world, **body_conf, position=position)
        bodies.append(body)
    return bodies, free_positions.difference(spawn_position_ids)

