from typing import Any, Callable, Optional, Union, Tuple, Dict, Set, List

import numpy as np

import pygame

from Box2D import ( # type: ignore
    b2World, b2Body, b2CircleShape, b2PolygonShape, b2EdgeShape, b2Vec2, 
    b2Transform, b2Mat22, b2Fixture)

import masurvival.simulation as sim
from masurvival.semantics import Agents, Boxes

Color = pygame.Color

# each artist is usually associated to a simulation module
class Artist:
    def draw(self, canvas: 'Canvas'):
        pass

default_background = Color('white')
default_outline = Color('gray25')

class Canvas:

    render_mode: str
    fps: int
    to_screen: b2Transform
    scale: b2Mat22
    surfaces: List[pygame.surface.Surface]
    window: pygame.surface.Surface
    clock: Optional[pygame.time.Clock] = None
    background: Color
    artists: List[Artist] = []

    def __init__(
            self, width: int, height: int, world_size: float,
            render_mode: str = 'human', layers: int = 2, fps: int = 30, 
            background: Color = default_background,
            artists: List[Artist] = []):
        if render_mode not in ['human', 'rgb_array']:
            raise ValueError(f'Invalid render mode "{render_mode}".')
        self.render_mode = render_mode
        self.fps = fps
        self.background = background
        self.to_screen = b2Transform()
        self.to_screen.Set(position=(width/2., height/2.), angle=0.0)
        self.scale = b2Mat22(width/world_size, 0., 0., -height/world_size)
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
        else: # render_mode == 'rgb_array'
            self.window = pygame.Surface((width, height))
        self.surfaces = [pygame.Surface((width, height), pygame.SRCALPHA)
                         for _ in range(layers)]
        self.artists += artists

    def close(self):
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()

    def clear(self):
        for i, surface in enumerate(self.surfaces):
            if i == 0:
                surface.fill(self.background)
            else:
                surface.fill(pygame.Color([0, 0, 0, 0]))        

    def render(self) -> np.ndarray:
        for artist in self.artists:
            artist.draw(self)
        for surface in self.surfaces:
            self.window.blit(surface, surface.get_rect())
        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps) # type: ignore
        pixels = pygame.surfarray.pixels3d(self.window)
        transposed_img = np.array(pixels)
        return np.transpose(transposed_img, axes=(1, 0, 2))

    def draw_segment(
            self, a: b2Vec2, b: b2Vec2, color: Color, layer: int = 0, 
            width: int = 1):
        surface = self.surfaces[layer]
        a = self.to_screen*(self.scale*a)
        b = self.to_screen*(self.scale*b)
        pygame.draw.line(surface, color, a, b, width=width) 

    def draw_dot(
            self, p: b2Vec2, color: Color, layer: int = 0, size: int = 1):
        surface = self.surfaces[layer]
        p = self.to_screen*(self.scale*p)
        pygame.draw.circle(surface, color, center=p, radius=size)

    def draw_shape(
            self, shape: Callable, fill: Optional[Color] = None,
            outline: Optional[Color] = None, layer: int = 0, **kwargs):
        surface = self.surfaces[layer]
        if fill is not None:
            shape(surface, color=fill, **kwargs)
        if outline is not None:
            shape(surface, color=outline, width=1, **kwargs)
        
    def draw_circle(
            self, radius: float, transform: b2Transform, 
            fill: Optional[Color] = None, outline: Optional[Color] = None, 
            layer: int = 0):
        center_world = transform*b2Vec2(0., 0.)
        center = self.to_screen*(self.scale*center_world)
        radius = (self.scale*b2Vec2(radius, 0.))[0]
        self.draw_shape(pygame.draw.circle, fill, outline, layer,
                        center=center, radius=radius)

    def draw_polygon(
            self, vertices: List[sim.Vec2], transform: b2Transform, 
            fill: Optional[Color] = None,
            outline: Optional[Color] = None, layer: int = 0):
        world_vertices = [transform*p for p in vertices]
        screen_vertices = [self.to_screen*(self.scale*p)
                           for p in world_vertices]
        self.draw_shape(pygame.draw.polygon, fill, outline, layer,
                        points=screen_vertices)

    def draw_fixture(
            self, fixture: b2Fixture, transform: b2Transform,
            fill: Optional[Color] = None, outline: Optional[Color] = None, 
            layer: int = 0):
        shape = fixture.shape
        args = (transform, fill, outline, layer)
        if isinstance(shape, b2CircleShape):
            self.draw_circle(shape.radius, *args)
        elif isinstance(shape, b2PolygonShape):
            self.draw_polygon(shape.vertices, *args)
        elif isinstance(shape, b2EdgeShape):
            color = fill or outline
            if color is None:
                return
            a, b = shape.vertices
            a, b = transform*a, transform*b
            self.draw_segment(a, b, color, layer)
        else:
            ValueError(f'draw_fixture: unsupported shape {type(shape)}')

    def draw_body(
            self, body: b2Body,
            fill: Optional[Union[Color, List[Color]]] = None,
            outline: Optional[Union[Color, List[Color]]] = None,
            layer: int = 0):
        transform = body.transform
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
            self.draw_fixture(fixture, transform, fill_, outline_, layer)


# concrete artists for concrete modules

class AgentsArtist(Artist):
    
    agents: Agents
    layer: int
    lidar_layer: int
    fill: Optional[Color]
    outline: Optional[Color]
    lidar_on: Optional[Color]
    lidar_off: Optional[Color]
    lidar_fill: Optional[Color]
    lidar_outline: Optional[Color]
    
    default_config: Dict[str, Any] = {
        'lidar_layer': 1,
        'fill': Color('cornflowerblue'),
        'outline': default_outline,
        'lidar_on': Color('indianred2'),
        'lidar_off': Color('gray'),
        'lidar_fill': None,
        'lidar_outline': Color('indianred2'),
    }
    
    def __init__(
            self, agents: Agents, layer: int = 0, lidar_layer: int = 0,
            fill: Optional[Color] = None, outline: Optional[Color] = None, 
            lidar_on: Optional[Color] = None,
            lidar_off: Optional[Color] = None,
            lidar_fill: Optional[Color] = None,
            lidar_outline: Optional[Color] = None):
        self.agents = agents
        self.layer = layer
        self.lidar_layer = lidar_layer
        self.fill = fill
        self.outline = outline
        self.lidar_on = lidar_on
        self.lidar_off = lidar_off
        self.lidar_fill = lidar_fill
        self.lidar_outline = lidar_outline
    
    def draw(self, canvas: Canvas):
        for body, sensor in zip(self.agents.bodies, self.agents.sensors):
            self.draw_agent(canvas, body, sensor)

    def draw_agent(self, canvas: Canvas, body: b2Body, lidar: sim.Lidar):
        canvas.draw_body(body, self.fill, self.outline, self.layer)
        origin, endpoints = lidar.origin, lidar.endpoints
        for endpoint, scan in zip(endpoints, lidar.observation):
            is_on = scan is not None
            laser_color = self.lidar_on if is_on else self.lidar_off
            if laser_color is None:
                continue
            length = 1 if not is_on else scan[1] # type: ignore
            endpoint = (1-length)*origin + length*endpoint
            canvas.draw_segment(
                origin, endpoint, laser_color, self.lidar_layer)
            if is_on:
                fixture = scan[0] # type: ignore
                transform = fixture.body.transform
                canvas.draw_fixture(
                    fixture, transform, self.lidar_fill, self.lidar_outline, 
                    self.lidar_layer)

class BoxesArtist(Artist):

    boxes: Boxes
    layer: int
    fill: Optional[Color]
    outline: Optional[Color]
    
    default_config: Dict[str, Any] = {
        'fill': Color('gold'),
        'outline': Color('gray25'),
    }
    
    def __init__(
            self, boxes: Boxes, layer: int = 0, fill: Optional[Color] = None, 
            outline: Optional[Color] = None):
        self.boxes = boxes
        self.layer = layer
        self.fill = fill
        self.outline = outline

    def draw(self, canvas: Canvas):
        for body in self.boxes.bodies:
            canvas.draw_body(body, self.fill, self.outline, self.layer)

