from typing import Optional, Tuple, List, Callable, Any, Dict

from Box2D import ( # type: ignore
    b2World, b2ContactListener, b2Body, b2Fixture, b2FixtureDef, b2Shape, 
    b2PolygonShape, b2Joint, b2RayCastCallback, b2Vec2, b2Mat22, b2Transform, 
    b2_staticBody, b2_dynamicBody)

import masurvival.geometry as geo


# stepping a simulation

def simulate(
        world: b2World, substeps: int, time_step: float = 1./60,
        velocity_iterations: int = 10, position_iterations: int = 10) \
        -> None:
    for _ in range(substeps):
        world.Step(time_step, velocity_iterations, position_iterations)
    world.ClearForces()


# creating and populating worlds

def empty_flatland() -> b2World:
    world = b2World(
        gravity=(0, 0), doSleep=True, contactListener=ContactListener())
    return world

# catches contacts between fixtures and calls their callbacks if set
class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)
    def BeginContact(self, contact):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        #print('beg contact')
        begin_contact_A = fixtureA.userData.get('BeginContact', None)
        begin_contact_B = fixtureB.userData.get('BeginContact', None)
        if begin_contact_A is not None:
            #print('beg contact A')
            begin_contact_A(fixtureA, fixtureB)
        if begin_contact_B is not None:
            #print('beg contact B')
            begin_contact_B(fixtureB, fixtureA)
    def EndContact(self, contact):
        #print('end contact')
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        end_contact_A = fixtureA.userData.get('EndContact', None)
        end_contact_B = fixtureB.userData.get('EndContact', None)
        if end_contact_A is not None:
            #print('end contact A')
            end_contact_A(fixtureA, fixtureB)
        if end_contact_B is not None:
            #print('end contact B')
            end_contact_B(fixtureB, fixtureA)
    def PreSolve(self, contact, oldManifold):
        #print('end contact')
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        pre_solve_A = fixtureA.userData.get('PreSolve', None)
        pre_solve_B = fixtureB.userData.get('PreSolve', None)
        if pre_solve_A is not None:
            pre_solve_A(fixtureA, fixtureB)
        if pre_solve_B is not None:
            pre_solve_B(fixtureB, fixtureA)
        categoryA = fixtureA.filterData.categoryBits
        categoryB = fixtureB.filterData.categoryBits
        maskA = fixtureA.filterData.maskBits
        maskB = fixtureB.filterData.maskBits
        contact.enabled = (maskA & categoryB) != 0 \
                          and (categoryA & maskB) != 0
    def PostSolve(self, contact, impulse):
        pass

ContactCallback = Callable[[b2Fixture, b2Fixture], Any]

# shape is square of side 1 if None
# copies userData to prevent object aliasing when using confs
# see below for the meaning of elevation
# ramp_edge, if given, makes the body behave like a ramp: it raises elevation 
# of bodies that cross that edge, and decreases their elevation when they 
# leave the ramp; for now, ramps must be polygons and sensor must be False
#TODO split this into several functions for body categories
def add_body(world: b2World, shape: b2Shape, position: geo.Vec2 = (0.,0.),  
             angle: float = 0., elevation: int = 0, extruded: bool = False,
             ramp_edge: Optional[int] = None, dynamic: bool = True,
             sensor: bool = False,
             begin_contact: Optional[ContactCallback] = None,
             end_contact: Optional[ContactCallback] = None,
             pre_solve: Optional[ContactCallback] = None,
             density: float = 1., restitution: float = 0.,
             damping: float = 0.5, userData: Dict[str, Any] = {}) -> b2Body:
    type = b2_dynamicBody if dynamic else b2_staticBody
    if ramp_edge is not None \
       and (sensor or not isinstance(shape, b2PolygonShape)):
        raise ValueError('Ramp bodies must be polygonal non-sensor bodies.')
    fixture = b2FixtureDef(shape=shape, density=density,
                           restitution=restitution, isSensor=sensor)
    fixture.userData = {}
    if begin_contact is not None:
        fixture.userData['BeginContact'] = begin_contact
    if end_contact is not None:
        fixture.userData['EndContact'] = end_contact
    if pre_solve is not None:
        fixture.userData['PreSolve'] = pre_solve
    # Make sure every body has a different dict object as userData.
    userData = dict(userData)
    body = world.CreateBody(
        type=type, position=position, angle=angle, fixtures=fixture,
        linearDamping=damping, angularDamping=damping, userData=userData)
    set_elevation(elevation, body)
    set_extruded(extruded, body)
    if ramp_edge is not None:
        vertices = body.fixtures[0].shape.vertices
        a, b = vertices[ramp_edge], vertices[(ramp_edge+1) % len(vertices)]
        edge = body.CreateEdgeFixture(vertices=[a, b])
        # Raise elevation when the body crosses the edge, but drop it only 
        # once it leaves the whole ramp fixture.
        edge.userData = {'PreSolve': lambda _, fB: on_stage(fixture, fB)}
        fixture.userData['EndContact'] = off_stage
    return body


# simulating discrete elevation

# the elevation level of a body makes it so that only bodies on the same level 
# can collide with it; only 15 levels are supported because of Box2D 
# limitations (so elevation must be in [0,14)); additionally, "extruded" 
# bodies can collide with bodies at eny elevation level, as if they extended 
# across all elevations (getting the elevation of an extruded body returns an 
# undefined int, while setting it has no effect)

# elevation is implemented using Box2D categories: bits 0-14 are for elevation 
# levels, and bit 15 is for extruded bodies
# for extrusion to work properly, extruded bodies need to have only bit 15 for 
# category bits, and all bits set for the mask; other bodies should also have 
# bit 15 always set in their mask, and never set in their category

# if the body is extruded, returns 15
# NOTE: uses the first fixture, ignoring the others
def get_elevation(body: b2Body) -> int:
    bits = body.fixtures[0].filterData.categoryBits
    # Get the index of the least significant bit.
    elevation = (bits & -bits).bit_length() - 1
    return elevation

# NOTE: if the body is extruded, invalid elevations *DON'T* raise errors
def set_elevation(elevation: int, body: b2Body):
    # If the body is extruded, ignore the operation.
    if get_extruded(body):
        #category_bits |= (1 << 15)
        return 
    if elevation < 0 or elevation > 14:
        raise ValueError(
            f'Elevation must be between 0 and 14, but is {elevation}.')
    category_bits = 0x0000 | (1 << elevation)
    for fixture in body.fixtures:
        fixture.filterData.categoryBits = category_bits
        # Add the category for extruded bodies to all bodies.
        fixture.filterData.maskBits = (1 << 15) | category_bits

def lift(body: b2Body, levels: int = 1) -> int:
    elevation = get_elevation(body)
    new_elevation = elevation + levels
    set_elevation(new_elevation, body)
    return new_elevation

def drop(body: b2Body, levels: int = 1) -> int:
    elevation = get_elevation(body)
    new_elevation = elevation - levels
    set_elevation(new_elevation, body)
    return new_elevation

# NOTE: uses fixture 0, ignoring the others
def get_extruded(body: b2Body) -> bool:
    category = body.fixtures[0].filterData.categoryBits
    return bool(category & (1 << 15))

#TODO test these also after body initialization

# elevation is used when extruded is False
# NOTE: uses only fixture 0 to compute the new category bits, which will be 
# then set for all fixtures
def set_extruded(extruded: bool, body: b2Body, elevation: int = 0):
    category = body.fixtures[0].filterData.categoryBits
    mask = body.fixtures[0].filterData.maskBits
    if extruded:
        category = 0x0000 | (1 << 15)
        mask = 0xFFFF
    else:
        category = 0x0000
        mask = 0x0000
    for fixture in body.fixtures:
        fixture.filterData.categoryBits = category
        fixture.filterData.maskBits = mask
    if not extruded:
        set_elevation(elevation, body)

# mark a body as extruded
def extrude(body: b2Body):
    set_extruded(True, body)

# mark a body as non-extruded and give it an elevation
def steamroll(body: b2Body, elevation: int = 0):
    set_extruded(False, body, elevation=elevation)

# "stage" fixtures can be used to make bodies elevated temporarily, e.g. while 
# they are in contact; they also remember which bodies are on stage in their 
# userData
# these are used to implement ramps
# to create an elevator, one can feed the two functions below as contact 
# callbacks to the add_body function (along with giving sensor = True)

def on_stage(stage: b2Fixture, elevee: b2Fixture):
    if 'on_stage' not in stage.userData:
        stage.userData['on_stage'] = []
    on_stage = stage.userData['on_stage']
    if elevee in on_stage:
        return
    #print(f'and now: {elevee.body.userData["tag"]}')
    lift(elevee.body)
    #TODO also add this info to the body userData?
    on_stage.append(elevee)

def off_stage(stage: b2Fixture, elevated: b2Fixture):
    on_stage = stage.userData.get('on_stage', None)
    if on_stage is None or len(on_stage) == 0:
        return
    try:
        on_stage.remove(elevated)
    except ValueError:
        # Don't try to drop a body that was not lifted.
        return
    #print(f'you heard: {elevated.body.userData["tag"]}')
    drop(elevated.body)


# changing the properties of bodies

def set_static(body: b2Body):
    body.type = b2_staticBody

def set_dynamic(body: b2Body):
    body.type = b2_dynamicBody

def holding_joint(holder: b2Body, held: b2Body, world: b2World) -> b2Joint:
    axis = holder.transform*b2Vec2(1.,0.)
    #midpoint = 0.5*(holder.worldCenter + held.worldCenter)
    joint = world.CreateWeldJoint(
        bodyA=holder, bodyB=held, anchor=holder.worldCenter)
    return joint


# applying impulses

def apply_force(force: geo.Vec2, body: b2Body) -> None:
    body.ApplyForceToCenter(force, True)

def apply_impulse(impulse: geo.Vec2, body: b2Body) -> None:
    body.ApplyLinearImpulse(impulse, body.worldCenter, True)

def apply_angular_impulse(impulse: geo.Vec2, body: b2Body) -> None:
    body.ApplyAngularImpulse(impulse, True)


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
