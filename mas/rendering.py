from typing import Any, Callable, Optional, Union, Tuple, Dict, Set

import numpy as np

#TODO make this optional
import pygame

from Box2D import ( # type: ignore
    b2World, b2Body, b2CircleShape, b2PolygonShape, b2Vec2, b2Transform,
    b2Mat22)

import mas.geometry as geo
import mas.simulation as simulation


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int], pygame.Color]


# only draws the first fixture
def draw_body(canvas: pygame.Surface, body: b2Body, color: Optional[Color],
              to_screen: b2Transform, scale: b2Mat22,
              outline_color: Optional[Color] = None) -> None:
    fill_args = color and {'width': 0, 'color': color}
    outline_args = outline_color and {'width': 1, 'color': outline_color}
    shape = body.fixtures[0].shape
    shape_args: Dict
    draw_func: Callable
    if isinstance(shape, b2CircleShape):
        draw_func = pygame.draw.circle
        center_world = body.transform*b2Vec2(0., 0.)
        center_screen = to_screen*(scale*center_world)
        radius_screen = (scale*b2Vec2(shape.radius, 0.))[0]
        shape_args = {'center': center_screen, 'radius': radius_screen}
    elif isinstance(shape, b2PolygonShape):
        draw_func = pygame.draw.polygon
        local_vertices = shape.vertices
        to_world = body.transform
        world_vertices = [to_world*p for p in local_vertices]
        screen_vertices = [to_screen*(scale*p) for p in world_vertices]
        shape_args = {'points': screen_vertices}
    else:
        raise ValueError(f'draw_body: unsupported shape {shape}')
    if fill_args is not None:
        assert(isinstance(fill_args, dict))
        draw_func(canvas, **shape_args, **fill_args)
    if outline_args is not None:
        assert(isinstance(outline_args, dict))
        draw_func(canvas, **shape_args, **outline_args)


def draw_world(canvas: pygame.Surface, world: b2World, world_size: float,
               colors: Dict[str, Color],
               outline_colors: Dict[str, Color] = {}) -> None:
    default_color = colors.get('default', None)
    default_outline_color = outline_colors.get('default', None)
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    for body in world:
        color = colors.get(body.userData, default_color)
        outline_color = outline_colors.get(body.userData, 
                                           default_outline_color)
        draw_body(canvas, body, color, to_screen, scale, 
                  outline_color=outline_color)


def draw_points(canvas, world_xs: np.ndarray, world_ys: np.ndarray,
                world_size: float, active_points: Set[int],
                active_color: Color, inactive_color: Color,
                rect_side: int = 4) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    for p_id in range(world_xs.shape[0]):
        color = active_color if p_id in active_points else inactive_color
        p_world = b2Vec2(world_xs[p_id], world_ys[p_id])
        p_screen = to_screen*(scale*p_world)
        rect = pygame.Rect(p_screen[0]-rect_side/2, p_screen[1]-rect_side/2, 
                           rect_side, rect_side)
        pygame.draw.rect(canvas, color, rect)


def draw_ray(canvas: pygame.Surface, world_size: float,
             transform: b2Transform, angle: float, depth: float,
             color: Color) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    start_world = transform*b2Vec2(0.,0.)
    start_screen = to_screen*(scale*start_world)
    ray_world = geo.from_polar(length=depth, angle=angle)
    end_world = transform*ray_world
    end_screen = to_screen*(scale*end_world)
    pygame.draw.line(canvas, color, start_screen, end_screen)


def draw_laser(
        canvas: pygame.Surface, angle: float, radius: float,
        transform: b2Transform, to_screen: b2Transform, scale: b2Mat22, 
        start_screen: geo.Vec2, scan: Optional[simulation.LaserScan],
        on_color: Color, off_color: Optional[Color] = None,
        scanned_outline_color: Optional[Color] = None) -> None:
    scanned_outline_color = scanned_outline_color or on_color
    laser_world = geo.from_polar(length=radius, angle=angle)
    end_world = transform*laser_world
    end_screen = to_screen*(scale*end_world)
    color = off_color
    mid_screen: Optional[Tuple[float, float]] = None
    if scan is not None:
        color = on_color
        scanned_body = scan[0].body
        active_laser = geo.from_polar(length=scan[1], angle=angle)
        mid_world = transform*active_laser
        mid_screen = to_screen*(scale*mid_world)
        draw_body(canvas, body=scanned_body, to_screen=to_screen, 
                  scale=scale, color=None,
                  outline_color=scanned_outline_color)
    if mid_screen is not None:
        pygame.draw.line(canvas, on_color, start_screen, mid_screen)
        #pygame.draw.line(canvas, off_color, mid_screen, end_screen)
    elif off_color is not None:
        pygame.draw.line(canvas, off_color, start_screen, end_screen)


def draw_lidar(
        canvas: pygame.Surface, world_size: float, n_lasers: int,
        transform: b2Transform, angle: float, radius: float,
        scan: simulation.LidarScan, on_color: Color,
        off_color: Optional[Color] = None,
        scanned_outline_color: Optional[Color] = None) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    to_screen = b2Transform()
    to_screen.Set(position=(w/2., h/2.), angle=0.0)
    scale = b2Mat22(w/world_size, 0., 0., -w/world_size)
    start_world = transform*b2Vec2(0.,0.)
    start_screen = to_screen*(scale*start_world)
    for laser_id in range(n_lasers):
        laser_angle = laser_id*(angle/(n_lasers-1)) - angle/2.
        draw_laser(
            canvas, angle=laser_angle, radius=radius, transform=transform, 
            to_screen=to_screen, scale=scale, start_screen=start_screen, 
            scan=scan[laser_id], on_color=on_color, off_color=off_color, 
            scanned_outline_color=scanned_outline_color)

