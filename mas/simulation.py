from typing import Optional, Tuple, List

from Box2D import ( # type: ignore
    b2World, b2Body, b2Fixture, b2RayCastCallback, b2Vec2, b2Mat22, 
    b2Transform)

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

# observing the world

LidarScan = List[Optional[Tuple[b2Fixture, float]]]

def lidar_scan(world: b2World, n_lasers: int, transform: b2Transform,
               angle: float, radius: float) -> LidarScan:
    scanned: List[Optional[Tuple[b2Fixture, float]]] = []
    start = transform*b2Vec2(0.,0.)
    for laser_id in range(n_lasers):
        laser_scan = LaserRayCastCallback()
        laser_angle = angle*(laser_id/n_lasers - 0.5)
        laser = geo.from_polar(length=radius, angle=laser_angle)
        end = transform*laser
        world.RayCast(laser_scan, start, end)
        if laser_scan.relative_depth is not None:
            depth = radius*laser_scan.relative_depth
            assert(laser_scan.fixture is not None)
            scanned.append((laser_scan.fixture, depth))
        else:
            scanned.append(None)
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
