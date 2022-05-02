#TODO test if this import fails if pygame is not available even when this module is not explicitely imported, but any parent or sibling module were imported
import pygame

import multiagent_survival.engine as engine


def draw_world(world, canvas):
    w, h = canvas.get_width(), canvas.get_height()
    s = w / 20. # scale
    for body in world:
        local_vertices = body.fixtures[0].shape.vertices
        x, y = body.position
        world_vertices = [ (p[0] + x, -(p[1] + y)) for p in local_vertices]
        vertices = [(s*p[0]+w/2., s*p[1]+h/2.) for p in world_vertices]
        color = (255, 0, 0)
        pygame.draw.polygon(canvas, color, vertices)

