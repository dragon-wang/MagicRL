from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from magicrl.env.utils import gymnasium_step_type
from magicrl.env.worker import BaseEnvWorker


class DummyEnvWokrder(BaseEnvWorker):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.result = None

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)

    def recv(self) -> Union[Tuple[Any, dict], gymnasium_step_type]:
        return self.result

    def seed(self, seed: int) -> None:
        self.action_space.seed(seed=seed)
        self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()
    
    def close_env(self) -> Any:
        return self.env.close()
