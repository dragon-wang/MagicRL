from typing import Any, Optional, Tuple, Union, Dict

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

    def evaluate(self) -> Dict:
        episode_rewards = np.zeros(self.env_num, dtype=np.float32)
        episode_lengths = np.zeros(self.env_num, dtype=np.int32)

        obs, _ = self.env.reset()
        while True:
            act = self.agent.select_action(obs, eval=True)
            obs, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)

            episode_rewards += reward * (1. - done)
            episode_lengths += 1 - done

            if np.all(done):
                break

        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        avg_length = np.mean(episode_lengths)
    
        evaluate_summaries = {'avg_reward': avg_reward, 
                              'max_reward': max_reward,
                              'min_reward': min_reward,
                              'avg_length': avg_length}
        
        print(f'Evaluate at train step: {self.agent.train_step}', 
               'average reward: {avg_reward}, avergae length: {avg_length} \n'
              f'Rewards of {self.env_num} envs: {episode_rewards}')

        return evaluate_summaries
        



            

        