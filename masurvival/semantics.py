from typing import Any, Type, Optional, Tuple, List, Dict, Set, NamedTuple
from operator import attrgetter

from Box2D import ( # type: ignore
    b2Vec2, b2World, b2Body)

import masurvival.simulation as sim
import masurvival.worldgen as worldgen


# circles with lidars and dynamic motors
class Agents(sim.Module):
    
    spawner: worldgen.Spawner
    prototypes: List[sim.BodyPrototype] = []
    motors: List[sim.DynamicMotor] = []
    sensors: List[sim.Lidar] = []
    bodies: List[sim.b2Body] = []

    def __init__(
            self, n_agents: int, agent_size: float, lidar_n_lasers: int, 
            lidar_fov: float, lidar_depth: float, motor_linear_impulse: float, 
            motor_angular_impulse: float,
            spawner: Optional[worldgen.Spawner] = None):
        shape = sim.circle_shape(radius=agent_size/2)
        lidar_args = (lidar_n_lasers, lidar_fov, lidar_depth)
        motor_args = (motor_linear_impulse, motor_angular_impulse)
        for _ in range(n_agents):
            self.prototypes.append(sim.BodyPrototype(shape))
            self.sensors.append(sim.Lidar(*lidar_args))
            self.motors.append(sim.DynamicMotor(*motor_args))
        self.dependencies = self.sensors + self.motors # type: ignore
        if spawner is not None:
            self.spawner = spawner

    def reset(self, world: sim.b2World):
        n_agents = len(self.prototypes)
        placements = self.spawner.placements(n_agents)
        self.bodies = []
        for position, prototype, sensor, motor \
        in zip(placements, self.prototypes, self.sensors, self.motors):
            body = sim.spawn(world, prototype, position)
            self.bodies.append(body)
            sensor.body = body
            motor.body = body

class Boxes(sim.Module):

    spawner: worldgen.Spawner
    prototypes: List[sim.BodyPrototype]
    bodies: List[sim.b2Body]

    def __init__(
            self, n_boxes: int, box_size: float,
            spawner: Optional[worldgen.Spawner] = None):
        shape = sim.rect_shape(width=box_size, height=box_size)
        self.prototypes = [sim.BodyPrototype(shape)]*n_boxes
        if spawner is not None:
            self.spawner = spawner

    def reset(self, world: sim.b2World):
        n_boxes = len(self.prototypes)
        placements = self.spawner.placements(n_boxes)
        self.bodies = []
        for position, prototype in zip(placements, self.prototypes):
            body = sim.spawn(world, prototype, position)
            self.bodies.append(body)
