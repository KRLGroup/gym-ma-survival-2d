from Box2D import ( # type: ignore
    b2World, b2Body, b2Fixture, b2RayCastCallback)

from mas.types import Vec2


def simulate(
        world: b2World, substeps: int, time_step: float = 1./60,
        velocity_iterations: int = 10, position_iterations: int = 10) \
        -> None:
    for _ in range(substeps):
        world.Step(time_step, velocity_iterations, position_iterations)
    world.ClearForces()

def apply_force(force: Vec2, body: b2Body) -> None:
    body.ApplyForceToCenter(force, True)

def apply_impulse(impulse: Vec2, body: b2Body) -> None:
    body.ApplyLinearImpulse(impulse, body.worldCenter, True)
