from typing import Any, Union, Optional, Tuple, List
import math

from Box2D import ( # type: ignore
     b2World, b2Body, b2FixtureDef, b2PolygonShape, b2CircleShape,
     b2_dynamicBody, b2_staticBody, b2Vec2, b2Shape)

import mas.geometry as geo


# shape is square of side 1 if None
def add_body(world: b2World, position: geo.Vec2 = (0.,0.), angle: float = 0.,
             shape: Optional[b2Shape] = None, dynamic: bool = True,
             density: float = 1., restitution: float = 0.,
             damping: float = 0.5, userData: Any = None) -> b2Body:
    type = b2_dynamicBody if dynamic else b2_staticBody
    shape = shape or square_shape(side=1.)
    fixture = b2FixtureDef(shape=shape, density=density,
                           restitution=restitution,)
    body = world.CreateBody(
        type=type, position=position, angle=angle, fixtures=fixture,
        linearDamping=damping, angularDamping=damping, userData=userData)
    return body


def square_shape(side: float) -> b2Shape:
    return b2PolygonShape(box=(side/2., side/2.))

def rect_shape(width: float, height: float) -> b2Shape:
    return b2PolygonShape(box=(width/2., height/2.))

def circle_shape(radius: float) -> b2Shape:
    return b2CircleShape(radius=radius)

# in these, relative_size is in [0.,1.] w.r.t. given world_size

_density = 1.
_restitution = 0.
_damping = 0.8

# relative size controls diameter
def agent_body_spec(world_size: float, relative_size: float = 0.05,
                    tag: str = 'agent'):
    return {
        'shape': circle_shape(radius=(world_size*relative_size)/2.),
        'dynamic': True,
        'density': 2.*_density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': tag,}

def box_body_spec(world_size: float, relative_size: float = 0.1,
                  movable: bool = True, tag: str = 'box'):
    return {
        'shape': square_shape(side=world_size*relative_size),
        'dynamic': movable,
        'density': _density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': tag,}

# horizontal wall; aspect_ratio is wall length / wall width
def wall_body_spec(world_size: float, aspect_ratio=100.,
                   relative_size: float = 1., tag: str = 'wall'):
    width = world_size*relative_size
    height = width/aspect_ratio
    return {
        'shape': rect_shape(width=width, height=height),
        'dynamic': False,
        'density': _density,
        'restitution': _restitution,
        'damping': _damping,
        'userData': tag,}


def populate_world(world: b2World, world_size: float) \
        -> Tuple[b2Body, b2Body, b2Body, List[b2Body]]:
    room_rel_size = 0.95
    wall_aspect_ratio = 100.
    # body specifications
    agent_spec = agent_body_spec(world_size)
    box_spec = box_body_spec(world_size)
    pillar_spec = box_body_spec(world_size, movable=False, tag='pillar')
    wall_spec = wall_body_spec(world_size, relative_size=room_rel_size, 
                               aspect_ratio=wall_aspect_ratio)
    # add bodies to the world
    agent = add_body(world, **agent_spec, position=(0., world_size/5.))
    box = add_body(world, **box_spec, position=(world_size/5.,0.))
    pillar = add_body(world, **pillar_spec, position=(0., 0.))
    walls = []
    walls_offset = room_rel_size*world_size/2. \
                   - (world_size/wall_aspect_ratio)/2
    wall_placements = [
        ((0., walls_offset), 0.), # north wall
        ((0., -walls_offset), 0.), # south wall
        ((walls_offset, 0.), math.pi/2.), # east wall
        ((-walls_offset, 0.), math.pi/2.),] # west wall
    for position, angle in wall_placements:
        wall = add_body(world, **wall_spec, position=position, angle=angle)
        walls.append(wall)
    return agent, box, pillar, walls
