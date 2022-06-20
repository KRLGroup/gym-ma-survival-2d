from typing import Type, Sequence, Optional, Tuple, List, Callable, Any, Dict, Union, NamedTuple

from Box2D import ( # type: ignore
    b2World, b2ContactListener, b2Body, b2Fixture, b2FixtureDef, b2Shape, 
    b2CircleShape,
    b2PolygonShape, b2Joint, b2RayCastCallback, b2Vec2, b2Mat22, b2Transform, 
    b2_staticBody, b2_dynamicBody)


# basic geometry

Vec2 = Union[Tuple[float, float], b2Vec2]
Vec3 = Tuple[float, float, float]

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


# creating bodies with modular behaviour

# A module is associated to a group of bodies, which in turn may have several 
# modules associated to it; each module may control the behaviour of bodies in 
# the group by defining one or more methods of the Module class.
# Each body always knows which group it belongs to; by accessing the group, 
# one can spawn and despawn bodies in that group and access their module-
# specific data (if any), so Box2D methods should not be used for that since 
# they mess up the group state.
# A more advanced case is when a module is used in more than one group; in 
# that case, it is its responsibility to distinguish between groups when its 
# methods are called.
# Things NOT to do inside module methods (these WILL mess up the group state):
# - change the list of bodies, modules or set the world attribute of the group
# - create or destroy a body that belongs in a group without using the group's 
#   spawn or despawn methods
# - change the userData of a body that belongs in a group
# Things that can be done safely:
# - perform raycasts on group.world
# - [...]
# Good examples of usage are in the semantics module.
class Module:
    # called when sim is reset
    def post_reset(self, group: 'Group'):
        pass
    # called before each sim step
    def pre_step(self, group: 'Group'):
        pass
    # called after each sim step
    def post_step(self, group: 'Group'):
        pass
    # called when new bodies spawn in the group (they wont yet be in the 
    # group.bodies)
    def post_spawn(self, bodies: List[b2Body]):
        pass
    # called when some bodies are about to despawn from the group
    def pre_despawn(self, bodies: List[b2Body]):
        pass


# utiltiy type to pack Box2D body definitions

default_density = 1.
default_restitution = 0.
default_damping = 0.8

class Prototype(NamedTuple):
    shape: b2Shape
    dynamic: bool = True
    density: float = default_density
    restitution: float = default_restitution
    damping: float = default_damping
    sensor: bool = False

# either position or position and orientation
Placement = Union[Vec3, Vec2]

class Group:

    bodies: List[b2Body]
    modules: Dict[Type, List[Module]]
    world: b2World

    @staticmethod
    def body_group(body: b2Body) -> 'Group':
        return body.userData

    @staticmethod
    def body_modules(body: b2Body) -> Dict[Type, List[Module]]:
        return Group.body_group(body).modules

    def __init__(self, modules: Sequence[Module] = []):
        self.bodies = []
        self.modules = {}
        self.add_modules(modules)
    
    def add_modules(self, modules: Sequence[Module]):
        [self.add_module(module) for module in modules]
    
    def add_module(self, module: Module):
        type_ = type(module)
        if type_ not in self.modules:
            self.modules[type_] = []
        self.modules[type_].append(module)

    def reset(self, world: b2World):
        #TODO destroy all bodies in the previous world?
        self.world = world
        self.bodies = []
        for modules in self.modules.values():
            for module in modules:
                module.post_reset(self)

    def pre_step(self):
        for modules in self.modules.values():
            for module in modules:
                module.pre_step(self)

    def post_step(self):
        for modules in self.modules.values():
            for module in modules:
                module.post_step(self)

    def spawn(self, prototypes: List[Prototype], placements: List[Placement]):
        bodies = [self._create_body(proto, pos)
                  for proto, pos in zip(prototypes, placements)]
        for modules in self.modules.values():
            for module in modules:
                module.post_spawn(bodies)
        self.bodies += bodies

    def despawn(self, bodies: List[b2Body]):
        for modules in self.modules.values():
            for module in modules:
                module.pre_despawn(bodies)
        self.bodies = [b for b in self.bodies if b not in bodies]
        [self._destroy_body(b) for b in bodies]

    def _create_body(self, prototype: Prototype, placement: Placement):
        has_angle = len(placement) == 3
        position = placement if not has_angle else placement[0:2]
        angle = 0 if not has_angle else placement[2] # type: ignore
        type = b2_dynamicBody if prototype.dynamic else b2_staticBody
        fixture = b2FixtureDef(
            shape=prototype.shape, density=prototype.density,
            restitution=prototype.restitution, isSensor=prototype.sensor)
        body = self.world.CreateBody(
            type=type, position=position, angle=angle, fixtures=fixture,
            linearDamping=prototype.damping, angularDamping=prototype.damping, 
            userData=self)
        return body

    def _destroy_body(self, body: b2Body):
        self.world.DestroyBody(body)


class Simulation:

    world: b2World
    substeps: int
    time_step: float
    velocity_iterations: int
    position_iterations: int
    groups: List[Group] = []

    def __init__(
            self, substeps: int = 2, time_step: float = 1/60, 
            velocity_iterations: int = 10, position_iterations: int = 10,
            groups: List[Group] = []):
        self.substeps = substeps
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.groups += groups

    def reset(self):
        self.world = b2World(gravity=(0, 0), doSleep=True)        
        for group in self.groups:
            group.reset(self.world)

    def step(self):
        for group in self.groups:
            group.pre_step()
        for _ in range(self.substeps):
            self.world.Step(
                self.time_step, self.velocity_iterations, 
                self.position_iterations)
        self.world.ClearForces()
        for group in self.groups:
            group.post_step()


# basic concrete modules

class Lidars(Module):

    n_lasers: int
    fov: float
    depth: float
    scans: List[List['LaserScan']] = []
    origins: List[b2Vec2] = []
    endpoints: List[List[b2Vec2]] = []

    def __init__(self, n_lasers: int, fov: float, depth: float):
        self.n_lasers = n_lasers
        self.fov = fov
        self.depth = depth

    def post_step(self, group: Group):
        self.origins = [body.position for body in group.bodies]
        orientations = [body.angle for body in group.bodies]
        self.endpoints = [self._endpoints(p, a)
                          for p, a in zip(self.origins, orientations)]
        self.scans = [[laser_scan(group.world, a, b) for b in bs]
                      for a, bs in zip(self.origins, self.endpoints)]

    def _endpoints(self, origin: b2Vec2, orientation: float):
        endpoints = []
        for i in range(self.n_lasers):
            angle = i*(self.fov/(self.n_lasers-1)) - self.fov/2.
            angle += orientation
            endpoint = origin + from_polar(length=self.depth, angle=angle)
            endpoints.append(endpoint)
        return endpoints


class DynamicMotors(Module):

    # (*linear,angular) impulses given for unitary controls
    impulse: Vec3
    # whether to avoid resetting the controls to 0 after each application
    drift: bool
    controls: List[Vec3] = []

    def __init__(self, impulse: Vec3, drift: bool = False):
        self.impulse = impulse
        self.drift = drift

    # completes missing controls with 0s, ignores excess controls
    def pre_step(self, group: Group):
        missing = len(group.bodies) - len(self.controls)
        if missing > 0:
            self.controls += [(0,0,0)]*missing
        for body, control in zip(group.bodies, self.controls):
            self._apply_control(body, control)
        if not self.drift:
            self.controls = []

    def _apply_control(self, body: b2Body, control: Vec3):
        R = body.transform.R
        p = body.worldCenter
        parallel_impulse = control[0]*self.impulse[0]
        normal_impulse = control[1]*self.impulse[1]
        linear_impulse = R*b2Vec2(parallel_impulse, normal_impulse)
        angular_impulse = control[2]*self.impulse[2]
        body.ApplyLinearImpulse(linear_impulse, p, True)
        body.ApplyAngularImpulse(angular_impulse, True)


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
