from typing import Any, Optional, Tuple, Union, List
import gymnasium as gym
import numpy as np
import torch

from magicrl.env.worker.base import EnvWorker
from magicrl.env.worker.dummy import DummyEnvWokrder
from magicrl.env.worker.subproc import SubprocEnvWorker


class BaseVectorEnv:
    def __init__(self, envs: List[gym.Env], worker_class: EnvWorker) -> None:
        self.workers = [worker_class(env) for env in envs]
        self.env_num = len(envs)
        self.action_spaces = [worker.action_space for worker in self.workers]
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return the number of environments."""
        return self.env_num
    
    def _wrap_id(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> Union[List[int], np.ndarray]:
        """Warp the given id.
        If the id is None, return all the id of envs
        """
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id
        
    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None, 
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the some envs and return initial observations and infos.

        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)

        for i in id:
            self.workers[i].send(None, **kwargs)
        reset_return_list = [self.workers[i].recv() for i in id]

        assert (
            isinstance(reset_return_list[0], (tuple, list)) and len(reset_return_list[0]) == 2
            and isinstance(reset_return_list[0][1], dict)
        ), "The environment does not adhere to the Gymnasium's API."

        obs_list = [rtn[0] for rtn in reset_return_list]
        infos_list = [rtn[1] for rtn in reset_return_list]

        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)

        return obs_stack, np.stack(infos_list)
    
    
    def step(self, action: Union[np.ndarray, List[np.ndarray]], id: Optional[Union[int, List[int], np.ndarray]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of some environments' with id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        assert len(action) == len(id)

        for i, j in enumerate(id):
            self.workers[j].send(action[i])

        result = []
    
        for j in id:
            step_return = self.workers[j].recv()
            step_return[-1]["env_id"] = j  # info["env_id"] = j
            result.append(step_return)

        obs_tuple, rew_tuple, term_tuple, trunc_tuple, info_tuple = tuple(zip(*result))

        try:
            obs_stack = np.stack(obs_tuple)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_tuple, dtype=object)
        return (
            obs_stack,
            np.stack(rew_tuple),
            np.stack(term_tuple),
            np.stack(trunc_tuple),
            np.stack(info_tuple),
        )
    
    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Set seeds for all the environments.
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]

        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        for worker, seed in zip(self.workers, seed_list):
            worker.seed(seed)
        
    def render(self):
        """Render all the environments.
        """
        return [woker.render() for woker in self.workers]

    def close(self) -> None:
        """Close all the environments.
        """
        self._assert_is_not_closed()
        for worker in self.workers:
            worker.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.
    """
    def __init__(self, envs: List[gym.Env]) -> None:
        super().__init__(envs, DummyEnvWokrder)


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.
    """
    def __init__(self, envs: List[gym.Env]) -> None:
        super().__init__(envs, SubprocEnvWorker)
    