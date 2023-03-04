# A 2D MARL survival environment

A Multi-Agent Reinforcement Learning survival environment in 2D with a
pseudo--gym interface.

![Interactive demo GIF](demo_gifs/interactive.gif)

See [here](https://arxiv.org/abs/2301.08030) for details on the
environment semantics.

## Installation

This is not a proper Python package yet; python source code is in the 
`masurvival` directory.

The code assumes that the version 3.9 of Python is used. The
dependencies are those listed in requirements.txt and PyBox2d
2.3.10, which can be installed with conda from the conda-forge channel.
More details on installing dependencies can be found below.

The environment itself (without rendering) only depends on Gym, Numpy
and PyBox2d. Rendering depends only on PyGame. The demo.py script
depends on pygifsicle and imageio for screenshot and gif recording.

### Installing dependencies

The most convenient way to install all dependencies is using conda,
which is needed to install PyBox2d anyway. It is convenient to create a
dedicated conda environment with

```
conda create -n env-name python=3.9
conda activate env-name
```

Since PyBox2d must be installed from conda-forge and gym fails to
install with the conda-forge installation of pip, the preferred way to
install all the dependencies using conda is to *append* the conda-forge
channel to the conda config, so that the defaults channel is used for
packages other than PyBox2d. This can be done with

```
conda config --append channels conda-forge
```

The packages in requirements.txt can then be installed using pip by
running

```
pip install -r requirements.txt
```

PyBox2d can then be installed with conda with

```
conda install pybox2d
```
