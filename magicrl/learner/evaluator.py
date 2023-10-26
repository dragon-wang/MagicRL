from typing import Any, Optional, Tuple, Union, Dict

from tqdm import tqdm
import gymnasium as gym
import numpy as np

from magicrl.env import BaseVectorEnv, DummyVectorEnv
from magicrl.data import BaseBuffer
from magicrl.agents import BaseAgent


class Evaluator:
    def __init__(self, env: Union[gym.Env, BaseVectorEnv], agent: BaseAgent) -> None:
        if isinstance(env, gym.Env):
            self.env = DummyVectorEnv([env])
        else:
            self.env = env
        self.agent = agent
        self.env_num = env.env_num

    # def evaluate(self, n_episode):
    #     episode_rewards = np.zeros(self.env_num, dtype=np.float32)
    #     episode_lengths = np.zeros(self.env_num, dtype=np.int32)
    #     for i in range(n_episode):
            
    def evaluate(self):
        episode_rewards = np.zeros(self.env_num, dtype=np.float32)
        episode_lengths = np.zeros(self.env_num, dtype=np.int32)
        dones = np.zeros(self.env_num, dtype=bool)

        obs, _ = self.env.reset()
        while True:
            act = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            episode_rewards += reward
            episode_lengths += 1



            

        