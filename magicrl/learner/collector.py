from typing import Any, Optional, Tuple, Union, Dict

import gymnasium as gym
import numpy as np

from magicrl.env import BaseVectorEnv
from magicrl.data import BaseBuffer
from magicrl.agents import BaseAgent


class Collector:
    def __init__(self, 
                 env: Union[gym.Env, BaseVectorEnv],
                 agent: BaseAgent, 
                 buffer: BaseBuffer
                 ) -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def collect(self,
                n_step: int = None,
                n_episode: int = None,
                random: bool = False
                ) -> Dict[str, Any]:
        pass
