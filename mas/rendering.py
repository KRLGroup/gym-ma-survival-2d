from typing import Union, Tuple, Dict

#TODO make this optional
import pygame

from Box2D import b2World, b2Transform, b2Mat22 # type: ignore


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def draw_world(world: b2World, canvas: pygame.Surface,
               colors: Dict[str, Color]) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    s = w / 20. # scale
    for body in world:
        local_vertices = body.fixtures[0].shape.vertices
        to_world = body.transform
        world_vertices = [ to_world*p for p in local_vertices]
        to_screen = b2Transform()
        to_screen.Set(position=(w/2., h/2.), angle=0.0)
        scale = b2Mat22(s, 0., 0., -s)
        vertices = [to_screen*(scale*p) for p in world_vertices]
        if body.userData in colors:
            color = colors[body.userData]
        else:
            color = (128, 128, 128)
        pygame.draw.polygon(canvas, color, vertices)

