from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import gym.spaces
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

# a wrapper that makes the MAS env look like a VecEnv, basically a modified version of DummyVecEnv
class VecEnvCostume(VecEnv):

    def __init__(self, env):
        self.env = env
        obs_space = gym.spaces.Dict({
            k: gym.spaces.Box(
                high=s.high[0], low=s.low[0], shape=s.shape[1:]
            )
            for k, s in env.observation_space.spaces.items()
        })
        assert isinstance(env.action_space, gym.spaces.Tuple)
        assert isinstance(
            env.action_space.spaces[0], gym.spaces.MultiDiscrete
        )
        action_space = env.action_space.spaces[0]
        VecEnv.__init__(self, env.n_agents, obs_space, action_space)
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, self.buf_rews, done, _ = self.env.step(self.actions)
        self.buf_dones[:] = done
        if done:
            # save final observation where user can get it, then reset
            for i in range(self.num_envs):
                self.buf_infos[i]["terminal_observation"] \
                    = {k: np.copy(v[i]) for k, v in obs.items()}
            obs = self.env.reset()
        self._save_obs(obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seed = env.seed(seed)
        return [seed]*self.num_envs

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        self._save_obs(obs)
        return self._obs_from_buf()

    def close(self) -> None:
        self.env.close()

    #TODO find out where this is called
    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError

    #TODO find out where this is called
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        return self.env.render(mode)

    def _save_obs(self, obs):
        for k, v in obs.items():
            self.buf_obs[k] = v

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    #TODO find out where these are used

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        raise NotImplementedError
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        raise NotImplementedError
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        raise NotImplementedError
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        raise NotImplementedError
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        raise NotImplementedError
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
