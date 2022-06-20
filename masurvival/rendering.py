from typing import Any, Callable, Optional, Union, Tuple, Dict, Set, List

import numpy as np

import pygame
import pygame.freetype

from Box2D import ( # type: ignore
    b2World, b2Body, b2CircleShape, b2PolygonShape, b2EdgeShape, b2Vec2, 
    b2Transform, b2Mat22, b2Fixture)

import masurvival.simulation as sim
import masurvival.semantics as sem

pygame.freetype.init()

Color = pygame.Color
Font = pygame.freetype.Font

default_font = pygame.freetype.SysFont('Roboto', size=12)

background = Color('white')

bodies_view_config = {
    'fill': Color('gold'),
    'outline': Color('gray25'),
    'layer': 0,
}

agent_bodies_view_config = {
    'fill': Color('cornflowerblue'),
    'outline': Color('gray25'),
    'layer': 0,
}

agent_lidars_view_config = {
    'on': Color('indianred2'),
    'off': Color('gray'),
    'fill': None,
    'outline': Color('indianred2'),
    'layer': 1,
}

walls_view_config = {
    'fill': Color('gray'),
    'outline': Color('gray25'),
    'layer': 0,
}

health_view_config = {
    'y_offset': 0.75,
    'fill': Color('green'),
    'outline': None,
    'layer': 1,
}

melee_view_config = {
    'on': Color('indianred2'),
    'off': Color('gray'),
    'fill': Color('indianred2'),
    'outline': Color('indianred2'),
    'layer': 1,
}

inventory_view_config = {
}


# each view should draw one aspect of the group (e.g. a module)
#TODO actually, these could very well be sim.Module's?
class View:
    def draw(self, canvas: 'Canvas', group: sim.Group):
        pass

class Canvas:

    render_mode: str
    fps: int
    to_screen: b2Transform
    scale: b2Mat22
    topleft: b2Vec2 # the top left corner of the canvas in world coords
    surfaces: List[pygame.surface.Surface]
    window: pygame.surface.Surface
    clock: Optional[pygame.time.Clock] = None
    background: Color
    views: Dict[sim.Group, List[View]] = {}

    def __init__(
            self, width: int, height: int, world_size: float,
            render_mode: str = 'human', layers: int = 2, fps: int = 30, 
            background: Color = background,
            views: Dict[sim.Group, List[View]] = {}):
        if render_mode not in ['human', 'rgb_array']:
            raise ValueError(f'Invalid render mode "{render_mode}".')
        self.render_mode = render_mode
        self.fps = fps
        self.background = background
        self.topleft = b2Vec2(-world_size/2, world_size/2)
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
        self.views |= views

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
        for group, views in self.views.items():
            for view in views:
                view.draw(self, group)
        for surface in self.surfaces:
            self.window.blit(surface, surface.get_rect())
        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps) # type: ignore
        pixels = pygame.surfarray.pixels3d(self.window)
        transposed_img = np.array(pixels)
        return np.transpose(transposed_img, axes=(1, 0, 2))

    def draw_text(
            self, text: str, position: sim.Vec2, font: Font = default_font, 
            foreground: Optional[Color] = None,
            background: Optional[Color] = None, layer: int = 0):
        surface = self.surfaces[layer]
        p = self.to_screen*(self.scale*position)
        font.render_to(
            surface, p, text, fgcolor=foreground, bgcolor=background)

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


# concrete views for semantics

class Bodies(View):

    fill: Optional[Color]
    outline: Optional[Color]
    layer: int

    def __init__(
            self, fill: Optional[Color] = None,
            outline: Optional[Color] = None, layer: int = 0):
        self.fill = fill
        self.outline = outline
        self.layer = layer

    def draw(self, canvas, group: sim.Group):
        for body in group.bodies:
            canvas.draw_body(body, self.fill, self.outline, self.layer)

class Lidars(View):

    on: Optional[Color]
    off: Optional[Color]
    fill: Optional[Color]
    outline: Optional[Color]
    layer: int

    def __init__(
            self, on: Optional[Color] = None, off: Optional[Color] = None, 
            fill: Optional[Color] = None, outline: Optional[Color] = None, 
            layer: int = 1):
        self.on = on
        self.off = off
        self.fill = fill
        self.outline = outline
        self.layer = layer

    def draw(self, canvas, group: sim.Group):
        for lidars in group.get(sim.Lidars):
            for origin, endpoints, scan \
            in zip(lidars.origins, lidars.endpoints, lidars.scans):
                self._draw_lidar(canvas, origin, endpoints, scan)

    def _draw_lidar(self, canvas, origin, endpoints, scans):
        for endpoint, scan in zip(endpoints, scans):
            is_on = scan is not None
            color = self.on if is_on else self.off
            if color is None:
                continue
            length = 1 if not is_on else scan[1] # type: ignore
            endpoint = (1-length)*origin + length*endpoint
            canvas.draw_segment(origin, endpoint, color, self.layer)
            if not is_on:
                continue
            fixture = scan[0] # type: ignore
            transform = fixture.body.transform
            canvas.draw_fixture(
                fixture, transform, self.fill, self.outline, self.layer)


# display inventories on the top left of the screen
class Inventory(View):
    
    font: Font
    foreground: Optional[Color]
    background: Optional[Color]
    layer: int
    offset: b2Vec2
    
    def __init__(
            self, font: Font = default_font,
            foreground: Optional[Color] = None,
            background: Optional[Color] = None, layer: int = 0):
        self.font = font
        self.foreground = foreground
        self.background = background
        self.layer = layer
        self.offset = b2Vec2(0.5, 0.5)
    
    def draw(self, canvas: Canvas, group: sim.Group):
        for m in group.get(sem.Inventory):
            for body, inv in m.inventories.items():
                canvas.draw_text(
                    f'{len(inv)}', body.position + self.offset, self.font, 
                    self.foreground, self.background, self.layer)
 

class Health(View):

    offset: b2Vec2
    fill: Optional[Color]
    outline: Optional[Color]
    layer: int

    def __init__(
            self, y_offset: float = 0, fill: Optional[Color] = None,
            outline: Optional[Color] = None, layer: int = 0):
        self.offset = b2Vec2(0, y_offset)
        self.fill = fill
        self.outline = outline
        self.layer = layer

    def draw(self, canvas: Canvas, group: sim.Group):
        for module in group.get(sem.Health):
            for body, health in module.healths.items():
                self._draw_health(body, health, canvas)

    def _draw_health(self, body: b2Body, health: int, canvas: Canvas):
        #TODO have params for the health bar dimensions
        healthbar = sim.rect_shape(health/10, 0.25)
        offset = b2Transform()
        offset.Set(position=(body.position + self.offset), angle=0)
        canvas.draw_polygon(
            healthbar.vertices, offset, self.fill, self.outline, self.layer)

class Melee(View):
    
    on: Optional[Color]
    off: Optional[Color]
    fill: Optional[Color]
    outline: Optional[Color]
    layer: int
    
    def __init__(
            self, on: Optional[Color] = None, off: Optional[Color] = None, 
            fill: Optional[Color] = None, outline: Optional[Color] = None, 
            layer: int = 0):
        self.on = on
        self.off = off
        self.fill = fill
        self.outline = outline
        self.layer = layer
    
    def draw(self, canvas: Canvas, group: sim.Group):
        for m in group.get(sem.Melee):
            for a, b, attack, target \
            in zip(m.origins, m.endpoints, m.attacks, m.targets):
                self._draw_melee(a, b, attack, target, canvas)

    def _draw_melee(
            self, a: b2Vec2, b: b2Vec2, attack: bool,
            target: Optional[b2Body], canvas: Canvas):
        color = self.on if attack else self.off
        if color is not None:
            canvas.draw_segment(a, b, color, self.layer)
        if target is None or not attack:
            return
        canvas.draw_body(target, self.fill, self.outline, self.layer)
           

