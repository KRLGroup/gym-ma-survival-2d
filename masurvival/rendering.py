from typing import Any, Callable, Optional, Union, Tuple, Dict, Set, List

import numpy as np

import pygame

from Box2D import ( # type: ignore
    b2World, b2Body, b2CircleShape, b2PolygonShape, b2EdgeShape, b2Vec2, 
    b2Transform, b2Mat22, b2Fixture)

import masurvival.geometry as geo
import masurvival.simulation as simulation


Color = pygame.Color
ColorKey = \
    Union[str,
          Tuple[int, int, int],
          Tuple[int, int, int, int],
          pygame.Color]

_default_color = Color('black')

class Palette:

    colors: Dict[str, Color] = {
        'default': Color('gold'),
        'default_outline': Color('gray25'),
        'background': Color('white'),
        'held': Color('sienna1'),
        'locked': Color('slateblue3'),
        'wall': Color('gray'),
        'pillar': Color('gray'),
        'agent': Color('cornflowerblue'),
        'ramp': Color('white'),
        'lidar_off': Color('gray'),
        'lidar_on': Color('indianred2'),
        'free_cell': Color('green'),
        'full_cell': Color('red'),
        'hand_on': Color('springgreen2'),
        'hand_off': Color('gray'),
        'ramp_edge': Color('red'),
        }

    def __init__(self, colors: Optional[Dict[str, Color]] = None):
        if colors is not None:
            self.colors = colors

    def get(self, key: ColorKey, default: Color = _default_color) -> Color:
        if isinstance(key, str):
            color = self.colors.get(key, None)
            if color is not None:
                return color
            try:
                color = Color(key)
            except ValueError:
                color = None
            if color is not None:
                return color
            return self.colors.get('default', _default_color)
        return Color(key)

    def __getitem__(self, key: ColorKey) -> Color:
        return self.get(key)


class Canvas:

    background: Color
    _render_mode: str
    _fps: int
    _to_screen: b2Transform
    _scale: b2Mat22
    _surfaces: List[pygame.surface.Surface]
    _window: pygame.surface.Surface
    _clock: Optional[pygame.time.Clock] = None
    
    def __init__(
            self, width: int, height: int, world_size: float,
            background: Color, render_mode: str = 'human',
            surfaces: int = 16, fps: int = 30):
        if render_mode not in ['human', 'rgb_array']:
            raise ValueError(f'Invalid render mode "{render_mode}".')
        self.background = background
        self._render_mode = render_mode
        self._fps = fps
        self._to_screen = b2Transform()
        self._to_screen.Set(position=(width/2., height/2.), angle=0.0)
        self._scale = b2Mat22(width/world_size, 0., 0., -height/world_size)
        if self._render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((width, height))
            self._clock = pygame.time.Clock()
        else: # render_mode == 'rgb_array'
            self._window = pygame.Surface((width, height))
        self._surfaces = [pygame.Surface((width, height), pygame.SRCALPHA)
                          for _ in range(surfaces)]

    def close(self):
        if self._render_mode == 'human':
            pygame.display.quit()
            pygame.quit()

    def clear(self):
        for i, surface in enumerate(self._surfaces):
            if i == 0:
                surface.fill(self.background)
            else:
                surface.fill(pygame.Color([0, 0, 0, 0]))

    def render(self) -> np.ndarray:
        for surface in self._surfaces:
            self._window.blit(surface, surface.get_rect())
        if self._render_mode == 'human':
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self._fps) # type: ignore
        pixels = pygame.surfarray.pixels3d(self._window)
        transposed_img = np.array(pixels)
        return np.transpose(transposed_img, axes=(1, 0, 2))

    def draw_segment(
            self, a: b2Vec2, b: b2Vec2, depth: int, color: Color,
            width: int = 1):
        surface = self._surfaces[depth]
        a = self._to_screen*(self._scale*a)
        b = self._to_screen*(self._scale*b)
        pygame.draw.line(surface, color, a, b, width=width) 

    def draw_dot(self, p: b2Vec2, depth: int, color: Color, size: int = 1):
        surface = self._surfaces[depth]
        p = self._to_screen*(self._scale*p)
        pygame.draw.circle(surface, color, center=p, radius=size)

    def draw_shape(
            self, shape: Callable, depth: int, fill: Optional[Color] = None, 
            outline: Optional[Color] = None, **kwargs):
        surface = self._surfaces[depth]
        if fill is not None:
            shape(surface, color=fill, **kwargs)
        if outline is not None:
            shape(surface, color=outline, width=1, **kwargs)
        
    def draw_circle(
            self, radius: float, transform: b2Transform, elevation: int, 
            fill: Optional[Color] = None, outline: Optional[Color] = None):
        center_world = transform*b2Vec2(0., 0.)
        center = self._to_screen*(self._scale*center_world)
        radius = (self._scale*b2Vec2(radius, 0.))[0]
        self.draw_shape(pygame.draw.circle, elevation, fill, outline,
                        center=center, radius=radius)

    def draw_polygon(
            self, vertices: List[geo.Vec2], transform: b2Transform, 
            elevation: int, fill: Optional[Color] = None,
            outline: Optional[Color] = None):
        world_vertices = [transform*p for p in vertices]
        screen_vertices = [self._to_screen*(self._scale*p)
                           for p in world_vertices]
        self.draw_shape(pygame.draw.polygon, elevation, fill, outline, 
                        points=screen_vertices)

    def draw_fixture(
            self, fixture: b2Fixture, transform: b2Transform, elevation: int, 
            fill: Optional[Color] = None, outline: Optional[Color] = None):
        shape = fixture.shape
        args = (transform, elevation, fill, outline)
        if isinstance(shape, b2CircleShape):
            self.draw_circle(shape.radius, *args)
        elif isinstance(shape, b2PolygonShape):
            self.draw_polygon(shape.vertices, *args)
        elif isinstance(shape, b2EdgeShape):
            a, b = shape.vertices
            a, b = transform*a, transform*b
            color = fill or outline or _default_color
            self.draw_segment(a, b, depth=elevation, color=color)
        else:
            ValueError(f'draw_fixture: unsupported shape {type(shape)}')


def draw_body(
        canvas: Canvas, body: b2Body,
        fill: Optional[Union[Color, List[Color]]],
        outline: Optional[Union[Color, List[Color]]],
        elevation: Optional[int] = None):
    if fill is None and outline is None:
        fill = _default_color
    transform = body.transform
    if elevation is None:
        elevation = simulation.get_elevation(body)
    for i, fixture in enumerate(body.fixtures):
        fill_: Optional[Color]
        if isinstance(fill, list):
            fill_ = fill[i]
        else:
            fill_ = fill
        outline_: Optional[Color]
        if isinstance(outline, list):
            outline_ = outline[i]
        else:
            outline_ = outline
        canvas.draw_fixture(fixture, transform, elevation, fill_, outline_)

def draw_laser(
        canvas: Canvas, origin: b2Vec2, angle: float, depth: float,
        scan: Optional[simulation.LaserScan], on: Color, off: Color, 
        elevation: int = 15):
    is_on = scan is not None
    color = on if is_on else off
    length = depth if not is_on else scan[1] # type: ignore
    endpoint = origin + geo.from_polar(length=length, angle=angle)
    canvas.draw_segment(a=origin, b=endpoint, depth=elevation, color=color)
    if is_on:
        body = scan[0].body # type: ignore
        draw_body(canvas, body, fill=None, outline=on, elevation=elevation)

def draw_lidar(
        canvas: Canvas, n_lasers: int, fov: float, radius: float,
        transform: b2Transform, scan: simulation.LidarScan, on: Color,
        off: Color, elevation: int = 15):
    origin = transform.position
    for i in range(n_lasers):
        angle = i*(fov/(n_lasers-1)) - fov/2.
        angle += transform.angle
        draw_laser(
            canvas, origin=origin, angle=angle, depth=radius, scan=scan[i], 
            on=on, off=off, elevation=elevation)

#TODO support fill and outline colors properly
def draw_world(canvas: Canvas, world: b2World, palette: Palette):
    default_fill = palette['default']
    default_outline = palette['default_outline']
    held = palette.get('held', default_fill)
    locked = palette.get('locked', default_fill)
    ramp_edge = palette.get('ramp_edge', default_fill)
    for body in world:
        tag = body.userData['tag']
        fill: Union[Color, List[Color]] = palette.get(tag, default_fill)
        outline = default_outline
        if held is not None and 'heldBy' in body.userData:
            fill = held
        if locked is not None and 'lockedBy' in body.userData:
            fill = locked
        if len(body.fixtures) == 2:
            assert(isinstance(fill, Color))
            fill = [fill, ramp_edge]
        draw_body(canvas, body, fill, outline)
