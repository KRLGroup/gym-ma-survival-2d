from typing import Optional, Tuple, List

from Box2D import ( # type: ignore
    b2World, b2Body, b2Fixture, b2Joint, b2RayCastCallback, b2Vec2, b2Mat22, 
    b2Transform, b2_staticBody, b2_dynamicBody)

import mas.geometry as geo


# stepping the simulation

def simulate(
        world: b2World, substeps: int, time_step: float = 1./60,
        velocity_iterations: int = 10, position_iterations: int = 10) \
        -> None:
    for _ in range(substeps):
        world.Step(time_step, velocity_iterations, position_iterations)
    world.ClearForces()


# acting on the world

def apply_force(force: geo.Vec2, body: b2Body) -> None:
    body.ApplyForceToCenter(force, True)

def apply_impulse(impulse: geo.Vec2, body: b2Body) -> None:
    body.ApplyLinearImpulse(impulse, body.worldCenter, True)

def apply_angular_impulse(impulse: geo.Vec2, body: b2Body) -> None:
    body.ApplyAngularImpulse(impulse, True)

def holding_joint(holder: b2Body, held: b2Body, world: b2World) -> b2Joint:
    axis = holder.transform*b2Vec2(1.,0.)
    joint = world.CreatePrismaticJoint(
        bodyA=holder, bodyB=held, anchor=holder.worldCenter,
        axis=axis, lowerTranslation=0.0, upperTranslation=0.0,
        enableLimit=True, enableMotor=False,)
    return joint

def set_static(body: b2Body):
    body.type = b2_staticBody

def set_dynamic(body: b2Body):
    body.type = b2_dynamicBody


# observing the world

LaserScan = Tuple[b2Fixture, float]

def laser_scan(world: b2World, transform: b2Transform, angle: float,
               depth: float) -> Optional[LaserScan]:
    start = transform*b2Vec2(0.,0.)
    laser = geo.from_polar(length=depth, angle=angle)
    end = transform*laser
    raycast = LaserRayCastCallback()
    world.RayCast(raycast, start, end)
    if raycast.relative_depth is None:
        return None
    depth = depth*raycast.relative_depth
    assert(raycast.fixture is not None)
    return raycast.fixture, depth

LidarScan = List[Optional[LaserScan]]

def lidar_scan(world: b2World, n_lasers: int, transform: b2Transform,
               angle: float, radius: float) -> LidarScan:
    scanned: List[Optional[Tuple[b2Fixture, float]]] = []
    start = transform*b2Vec2(0.,0.)
    for laser_id in range(n_lasers):
        laser_angle = laser_id*(angle/(n_lasers-1)) - angle/2.
        scan = laser_scan(world=world, transform=transform, 
                          angle=laser_angle, depth=radius)
        scanned.append(scan)
    return scanned

# from https://github.com/pybox2d/pybox2d/wiki/manual#ray-casts
# only retains the closest fixture found by the ray cast
class LaserRayCastCallback(b2RayCastCallback):
    fixture: Optional[b2Fixture] = None
    point: Optional[b2Vec2] = None
    normal: Optional[b2Vec2] = None
    relative_depth: Optional[float] = None
    def __init__(self) -> None:
        b2RayCastCallback.__init__(self)
    #TODO type hints
    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.relative_depth = fraction
        return fraction
