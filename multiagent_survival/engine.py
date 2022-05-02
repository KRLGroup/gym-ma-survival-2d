import numpy as np
from Box2D import b2World, b2FixtureDef, b2PolygonShape


class World:

    fixtures = {
      'square': b2FixtureDef(
          shape=b2PolygonShape(box=(0.5, 0.5)),
          density=1.0,
          restitution=0.0),}

    def __init__(self, time_step=1./60, velocity_iterations=10,
                 position_iterations=10):
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self._b2world = b2World(gravity=(0, 0), doSleep=True)
        self._bodies = []

    #TODO add option to make body static
    def add_dynamic_body(self, position, angle=0.0, fixture='square'):
        body = self._b2world.CreateDynamicBody(
            position=position, angle=angle, fixtures=self.fixtures[fixture])
        id = len(self._bodies)
        self._bodies.append(body)
        return id

    def apply_force(self, force, body_id):
        body = self._bodies[body_id]
        body.ApplyForceToCenter(force, True)

    def __getitem__(self, body_item):
        return self._bodies[body_item]

    def __iter__(self):
        return iter(self._bodies)

    def step(self, clear_forces=True):
        self._b2world.Step(self.time_step, self.velocity_iterations,
                           self.position_iterations)
        if clear_forces:
            self._b2world.ClearForces()

