from typing import Union, Tuple

from Box2D import b2Vec2, b2Mat22 # type: ignore

Vec2 = Union[Tuple[float, float], b2Vec2]

def from_polar(length: float, angle: float) -> b2Vec2:
    R = b2Mat22()
    R.angle = angle
    return R*b2Vec2(length, 0.)

