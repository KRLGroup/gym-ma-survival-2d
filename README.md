# Gym interface for a 2D MARL survival environment

A Multi-Agent Reinforcement Learning survival environment in 2D with a OpenAI 
Gym interface. This is not a proper Python package yet; source code is in the 
`masurvival` directory.

## Environment description

For now the semantics are almost the same as the `hide-and-seek-
environment-2d-version`: agents are circles which can move as unicycles and 
observe the environment with LIDAR sensors. They can also grab specific 
objects and lock/unlock them.

## Dependencies

- Box2D for physics

- Numpy for other math

- PyGame for rendering


