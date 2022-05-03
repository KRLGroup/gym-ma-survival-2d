from typing import Union, Tuple, Dict

#TODO make this optional
import pygame

from Box2D import b2World # type: ignore


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def draw_world(world: b2World, canvas: pygame.Surface,
               colors: Dict[str, Color]) -> None:
    w, h = canvas.get_width(), canvas.get_height()
    s = w / 20. # scale
    for body in world:
        local_vertices = body.fixtures[0].shape.vertices
        T = body.transform
        world_vertices = [ T*p for p in local_vertices]
        vertices = [(s*p[0]+w/2., -s*p[1]+h/2.) for p in world_vertices]
        if body.userData in colors:
            color = colors[body.userData]
        else:
            color = (128, 128, 128)
        pygame.draw.polygon(canvas, color, vertices)

