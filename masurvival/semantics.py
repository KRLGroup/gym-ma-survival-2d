from typing import Any, Type, Optional, Tuple, List, Dict, Set, NamedTuple
from operator import attrgetter

from Box2D import ( # type: ignore
    b2Vec2, b2World, b2Body)

import numpy as np

import masurvival.simulation as sim


def agent_prototype(agent_size: float) -> sim.Prototype:
    shape = sim.circle_shape(radius=agent_size/2)
    return sim.Prototype(shape=shape)

def box_prototype(box_size: float) -> sim.Prototype:
    shape = sim.rect_shape(width=box_size, height=box_size)
    return sim.Prototype(shape=shape)

def item_prototype(item_size: float):
    return sim.Prototype(sim.circle_shape(item_size/2), sensor=True)


# astract interface
class Spawner:
    def reset(self):
        pass
    def placements(self, n: int) -> List[b2Vec2]:
        return []

class SpawnGrid(Spawner):

    grid_size: int
    floor_size: float
    rng: np.random.Generator
    positions: List[b2Vec2]
    occupied: List[b2Vec2]
    
    def __init__(self, grid_size: int, floor_size: float):
        self.grid_size = grid_size
        self.floor_size = floor_size
    
    def reset(self):
        self.positions = square_grid(self.grid_size, self.floor_size)
        self.occupied = []
        self.rng.shuffle(self.positions)
    
    def placements(self, n: int) -> List[b2Vec2]:
        occupied = [self.positions.pop() for _ in range(n)]
        self.occupied += occupied
        return occupied


# spawns N bodies with given prototype on world reset
class ResetSpawns(sim.Module):
    
    def __init__(
            self, n_spawns: int, prototype: sim.Prototype, spawner: Spawner):
        self.n_spawns = n_spawns
        self.prototypes = [prototype]*self.n_spawns
        self.spawner = spawner
    
    def post_reset(self, group: sim.Group):
        group.spawn(self.prototypes, self.spawner.placements(self.n_spawns))

# spawns 4 immovable, thick walls enclosing a room of given size
class ThickRoomWalls(sim.Module):

    prototypes: List[sim.Prototype]
    placements: List[b2Vec2]

    def __init__(self, room_size: float, wall_aspect_ratio: float = 100):
        height = room_size
        width = height / wall_aspect_ratio
        shape = sim.rect_shape(width=width, height=height)
        self.prototypes = [sim.Prototype(shape=shape, dynamic=False)]*4
        offset = room_size/2
        self.placements = [
            (-offset, 0, 0), # west wall
            (0, offset, np.pi/2), # north wall
            (offset, 0, 0), # east wall
            (0, -offset, np.pi/2),] # south wall

    def post_reset(self, group: sim.Group):
        group.spawn(self.prototypes, self.placements)

# gives health to all bodies in its group and kills them when their health <= 
# 0
class Health(sim.Module):

    # starting health for all bodies with this module
    health: int
    healths: Dict[b2Body, int]

    def __init__(self, health: int):
        self.health = health

    def post_reset(self, group: sim.Group):
        self.healths = {body: self.health for body in group.bodies}

    def post_step(self, group: sim.Group):
        #TODO maybe optimize this by keeping the data in body.userData?
        for body in group.bodies:
            if body not in self.healths:
                self.healths[body] = self.health
        dead = [body for body, health in self.healths.items() if health <= 0]
        group.despawn(dead)

    def pre_despawn(self, bodies: List[b2Body]):
        for body in bodies:
            del self.healths[body]

    def damage(self, body: b2Body, damage: int):
        if body not in self.healths:
            return
        self.healths[body] -= damage

    def heal(self, body: b2Body, healing: int):
        if body not in self.healths:
            return
        self.healths[body] -= healing

# gives an short-range attack that damages the target (if it has an health 
# module)
#TODO also give cooldown?
class Melee(sim.Module):

    range: float
    damage: int
    drift: bool
    targets: List[Optional[b2Body]] = []
    origins: List[b2Vec2] = []
    endpoints: List[b2Vec2] = []
    attacks: List[bool] = []

    def __init__(self, range: float, damage: int, drift: bool = False):
        self.range = range
        self.damage = damage
        self.drift = drift

    def pre_step(self, group: sim.Group):
        missing = len(group.bodies) - len(self.attacks)
        if missing > 0:
            self.attacks += [False]*missing
        for body, target, attack \
        in zip(group.bodies, self.targets, self.attacks):
            if target is not None and attack:
                self._attack(target, self.damage)
        if not self.drift:
            self.attacks = []

    def post_step(self, group: sim.Group):
        self.origins = [body.position for body in group.bodies]
        hands = [sim.from_polar(length=self.range, angle=body.angle)
                 for body in group.bodies]
        self.endpoints = [a + d for a, d in zip(self.origins, hands)]
        scans = [sim.laser_scan(group.world, a, b)
                 for a, b in zip(self.origins, self.endpoints)]
        self.targets = [scan and scan[0].body for scan in scans]

    def _attack(self, target: b2Body, damage: int):
        healths = sim.Group.body_modules(target).get(Health, [])
        for health in healths:
            health.damage(target, damage) # type: ignore

# marks bodies in its group as items (can be picked up by inventories)
class Item(sim.Module):
    pass

# an inventory with automatic pickup of items in an AABB range around the 
# bodies of the group
class Inventory(sim.Module):

    range: float

    def __init__(self, range: float):
        self.range = range

    def post_step(self, group: sim.Group):
        r = self.range
        fixturess = [sim.aabb_query(group.world, body.position, r, r)
                     for body in group.bodies]
        candidatess = [[f.body for f in fixtures] for fixtures in fixturess]
        for body, candidates in zip(group.bodies, candidatess):
            for candidate in candidates:
                self._try_pickup(body, candidate)

    def _try_pickup(self, body: b2Body, candidate: b2Body):
        group = sim.Group.body_group(candidate)
        items = group.modules.get(Item, [])
        #TODO properly pick up the item instead of just despawning it
        if len(items) > 0:
            group.despawn([candidate])


# utilties

def square_grid(grid_size: int, floor_size: float) -> List[b2Vec2]:
    centers = np.arange(grid_size)/grid_size + 0.5/grid_size
    centers = floor_size*centers - floor_size/2.
    ii = np.arange(grid_size**2) % grid_size
    jj = np.arange(grid_size**2) // grid_size
    return [b2Vec2(centers[i], centers[j]) for i, j in zip(ii, jj)]


