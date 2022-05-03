from Box2D import (b2World, b2Body) # type: ignore

from mas.types import Vec2

def apply_force(force: Vec2, body: b2Body) -> None:
    body.ApplyForceToCenter(force, True)

def apply_impulse(impulse: Vec2, body: b2Body) -> None:
    body.ApplyLinearImpulse(impulse, body.worldCenter, True)

def step_world(world: b2World, time_step: float = 1./60,
               velocity_iterations: int = 10, position_iterations: int = 10,
               clear_forces: bool = True) -> None:
    world.Step(time_step, velocity_iterations, position_iterations)
    if clear_forces:
        world.ClearForces()
