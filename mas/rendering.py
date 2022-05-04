from typing import Union, Tuple, Dict

#TODO make this optional
import pygame

from Box2D import ( # type: ignore
    b2World, b2Body, b2CircleShape, b2PolygonShape, b2Vec2, b2Transform,
    b2Mat22)

import mas.geometry as geo
import mas.simulation as simulation


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


# only draws the first fixture
def draw_body(canvas: pygame.Surface, body: b2Body, color: Color,
              to_screen: b2Transform, scale: b2Mat22, width=0):
    shape = body.fixtures[0].shape
    if isinstance(shape, b2CircleShape):
        center_world = body.transform*b2Vec2(0., 0.)
        center_screen = to_screen*(scale*center_world)
        radius_screen = (scale*b2Vec2(shape.radius, 0.))[0]
        pygame.draw.circle(canvas, color, center_screen, radius_screen, 
                           width=width)
    elif isinstance(shape, b2PolygonShape):
        local_vertices = shape.vertices
        to_world = body.transform
        world_vertices = [ to_world*p for p in local_vertices]
        vertices = [to_screen*(scale*p) for p in world_vertices]
        pygame.draw.polygon(canvas, color, vertices, width=width)
    else:
        raise ValueError(f'draw_body: unsupported shape {shape}')


def draw_world(canvas: pygame.Surface, world: b2World, world_size: float,
               colors: Dict[str, Color]) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    for body in world:
        if body.userData in colors:
            color = colors[body.userData]
        else:
            color = (128, 128, 128)
        draw_body(canvas, body, color, to_screen, scale)
        

def draw_lidar(
      canvas: pygame.Surface, world_size: float, n_lasers: int,
      transform: geo.Vec2, angle: float, radius: float,
      scan: simulation.LidarScan, color: Color) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    start_world = transform*b2Vec2(0.,0.)
    start_screen = to_screen*(scale*start_world)
    for laser_id in range(n_lasers):
        laser_angle = angle*(laser_id/n_lasers - 0.5)
        laser_world = geo.from_polar(length=radius, angle=laser_angle)
        end_world = transform*laser_world
        end_screen = to_screen*(scale*end_world)
        color_ = color
        if scan[laser_id] is not None:
            color_ = (255 - color[0], 255 - color[1], 255 - color[2])
            scanned_body = scan[laser_id][0].body
            draw_body(canvas, scanned_body, color_, to_screen, scale, width=2)
        pygame.draw.line(canvas, color_, start_screen, end_screen)
