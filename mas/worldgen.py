from typing import Any, Union, Optional, Tuple

from Box2D import ( # type: ignore
     b2World, b2Body, b2FixtureDef, b2PolygonShape, b2_dynamicBody, 
     b2_staticBody, b2Vec2, b2Shape)

from mas.types import Vec2


def square_shape(side: float) -> b2Shape:
    return b2PolygonShape(box=(side/2., side/2.))


# shape is square of side 1 if None
def add_body(world: b2World, position: Vec2 = (0.,0.), angle: float = 0.,
             shape: Optional[b2Shape] = None, dynamic: bool = True,
             density: float = 1., restitution: float = 0.,
             damping: float = 0.5, userData: Any = None) -> b2Body:
    type = b2_dynamicBody if dynamic else b2_staticBody
    shape = shape or square_shape(side=1.)
    fixture = b2FixtureDef(shape=shape, density=density,
                           restitution=restitution),
    body = world.CreateBody(
        type=type, position=position, angle=angle, fixtures=fixture,
        linearDamping=damping, angularDamping=damping, userData=userData)
    return body

agent_body_spec = {
    'shape': square_shape(side=1.),
    'dynamic': True,
    'density': 1.,
    'restitution': 0.,
    'damping': 0.5,}

box_body_spec = {
    'shape': square_shape(side=1.5),
    'dynamic': False,
    'density': 1.,
    'restitution': 0.,
    'damping': 0.5,}

def populate_world(world: b2World) -> Tuple[b2Body, b2Body]:
    agent = add_body(**agent_body_spec, world=world, userData='agent')
    box = add_body(**box_body_spec, world=world, userData='box')
    return agent, box
