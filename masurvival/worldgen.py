from typing import Any, Union, Optional, Tuple, List, Dict, Set
import math

import numpy as np

from Box2D import ( # type: ignore
     b2World, b2Body, b2FixtureDef, b2PolygonShape, b2CircleShape,
     b2_dynamicBody, b2_staticBody, b2Vec2, b2Shape)


# astract interface
class Spawner:
    def placements(self, n: int) -> List[b2Vec2]:
        return []

class SpawnGrid(Spawner):

    positions: List[b2Vec2]
    
    def __init__(self, grid_size: int, floor_size: float):
        self.positions = square_grid(grid_size, floor_size)
    
    def shuffle(self, rng: np.random.Generator):
        rng.shuffle(self.positions)
    
    def placements(self, n: int) -> List[b2Vec2]:
        return [self.positions.pop() for _ in range(n)]


def square_grid(grid_size: int, floor_size: float) -> List[b2Vec2]:
    centers = np.arange(grid_size)/grid_size + 0.5/grid_size
    centers = floor_size*centers - floor_size/2.
    ii = np.arange(grid_size**2) % grid_size
    jj = np.arange(grid_size**2) // grid_size
    return [b2Vec2(centers[i], centers[j]) for i, j in zip(ii, jj)]
