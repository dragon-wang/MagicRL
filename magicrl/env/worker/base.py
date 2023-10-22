from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
from magicrl.env.utils import gymnasium_step_type
import gymnasium as gym
import numpy as np


class EnvWorker(ABC):
    "The code is based on tianshou's worker (https://github.com/thu-ml/tianshou/tree/master/tianshou/env/worker)"
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.is_closed = False
    
    @abstractmethod
    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        """Send an action signal to low-level worker.
        If action is none, send a 'reset' signal. Otherwise, send a 'step' signal.
        """
        pass
    
    @abstractmethod
    def recv(self) -> Union[Tuple[Any, dict], gymnasium_step_type]:
        """Receive the result of reset, step, render and close_env signal."""
        pass

    @abstractmethod
    def seed(self, seed: int) -> None:
        pass

    @abstractmethod
    def render(self) -> Any:
        pass

    @abstractmethod
    def close_env(self) -> Any:
        pass

    def close(self) -> Any:
        if self.is_closed:
            return
        self.is_closed = True
        return self.close_env()
    