from typing import Sequence, Optional, Tuple, List, Callable, Any, Dict, Union, NamedTuple

from Box2D import ( # type: ignore
    b2World, b2ContactListener, b2Body, b2Fixture, b2FixtureDef, b2Shape, 
    b2CircleShape,
    b2PolygonShape, b2Joint, b2RayCastCallback, b2Vec2, b2Mat22, b2Transform, 
    b2_staticBody, b2_dynamicBody)


# basic geometry

Vec2 = Union[Tuple[float, float], b2Vec2]

def from_polar(length: float, angle: float) -> b2Vec2:
    R = b2Mat22()
    R.angle = angle
    return R*b2Vec2(length, 0.)

def square_shape(side: float) -> b2Shape:
    return b2PolygonShape(box=(side/2., side/2.))

def rect_shape(width: float, height: float) -> b2Shape:
    return b2PolygonShape(box=(width/2., height/2.))

def circle_shape(radius: float) -> b2Shape:
    return b2CircleShape(radius=radius)


# body prototyping and spawning

default_density = 1.
default_restitution = 0.
default_damping = 0.8

BodyData = Union[int, float, str]

class BodyPrototype(NamedTuple):
    shape: b2Shape
    dynamic: bool = True
    density: float = default_density
    restitution: float = default_restitution
    damping: float = default_damping
    sensor: bool = False
    data: Dict[str, BodyData] = {}

def spawn(
        world: b2World, prototype: BodyPrototype,
        position: Vec2 = b2Vec2(0, 0), orientation: float = 0) -> b2Body:
    type = b2_dynamicBody if prototype.dynamic else b2_staticBody
    fixture = b2FixtureDef(
        shape=prototype.shape, density=prototype.density,
        restitution=prototype.restitution, isSensor=prototype.sensor)
    body = world.CreateBody(
        type=type, position=position, angle=orientation, fixtures=fixture,
        linearDamping=prototype.damping, angularDamping=prototype.damping, 
        userData=prototype.data)
    return body


# modules

class Module:
    dependencies: Sequence['Module'] = []
    def reset(self, world: b2World):
        pass
    def act(self, world: b2World):
        pass
    def observe(self, world: b2World):
        pass


# modular simulations

class Simulation:

    world: b2World
    substeps: int
    time_step: float
    velocity_iterations: int
    position_iterations: int
    modules: List[Module] = []

    def __init__(
            self, substeps: int = 2, time_step: float = 1/60, 
            velocity_iterations: int = 10, position_iterations: int = 10, 
            modules: List[Module] = []):
        self.substeps = substeps
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        [self.add_module(m) for m in modules]

    def add_module(self, module: Module):
        [self.add_module(d) for d in module.dependencies]
        self.modules.append(module)

    def reset(self):
        self.world = b2World(gravity=(0, 0), doSleep=True)        
        for module in self.modules:
            module.reset(self.world)
        for module in self.modules:
            module.observe(self.world)

    def step(self):
        for module in self.modules:
            module.act(self.world)
        for _ in range(self.substeps):
            self.world.Step(
                self.time_step, self.velocity_iterations, 
                self.position_iterations)
        self.world.ClearForces()
        for module in self.modules:
            module.observe(self.world)


# basic modules

class Lidar(Module):

    n_lasers: int
    fov: float
    depth: float
    body: b2Body
    origin: b2Vec2
    endpoints: b2Vec2
    observation: List['LaserScan'] = []

    def __init__(
            self, n_lasers: int, fov: float, depth: float,
            body: Optional[b2Body] = None):
        self.n_lasers = n_lasers
        self.fov = fov
        self.depth = depth
        if body is not None:
            self.body = body

    def observe(self, world: b2World):
        origin = self.body.position
        orientation = self.body.angle
        endpoints: b2Vec2 = []
        scan = []
        for i in range(self.n_lasers):
            angle = i*(self.fov/(self.n_lasers-1)) - self.fov/2.
            angle += orientation
            endpoint = origin + from_polar(length=self.depth, angle=angle)
            endpoints.append(endpoint)
            scan.append(laser_scan(world, origin, endpoint))
        self.observation = scan
        self.origin = origin
        self.endpoints = endpoints

class DynamicMotor(Module):

    linear_impulse: float
    angular_impulse: float
    drift: bool
    body: b2Body

    def __init__(
            self, linear_impulse: float, angular_impulse: float,
            body: Optional[b2Body] = None, drift: bool = False):
        self.linear_impulse = linear_impulse
        self.angular_impulse = angular_impulse
        if body is not None:
            self.body = body
        self.drift = drift

    def control(self, linear_control: Vec2, angular_control: float):
        self.linear_control = linear_control
        self.angular_control = angular_control

    def act(self, world: b2World):
        R = self.body.transform.R
        p = self.body.worldCenter
        parallel_impulse = self.linear_control[0]*self.linear_impulse
        normal_impulse = self.linear_control[1]*self.linear_impulse
        linear_impulse = R*b2Vec2(parallel_impulse, normal_impulse)
        angular_impulse = self.angular_control*self.angular_impulse
        self.body.ApplyLinearImpulse(linear_impulse, p, True)
        self.body.ApplyAngularImpulse(angular_impulse, True)
        if not self.drift:
            self.linear_control = b2Vec2(0, 0)
            self.angular_control = 0


# utilities

LaserScan = Optional[Tuple[b2Fixture, float]]

def laser_scan(world: b2World, start: Vec2, end: Vec2) -> LaserScan:
    start, end = b2Vec2(start), b2Vec2(end)
    raycast = LaserRayCastCallback()
    world.RayCast(raycast, start, end)
    if raycast.relative_depth is None:
        return None
    depth = raycast.relative_depth
    assert(raycast.fixture is not None)
    return raycast.fixture, depth

# utiltiy class from https://github.com/pybox2d/pybox2d/wiki/manual#ray-casts
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
