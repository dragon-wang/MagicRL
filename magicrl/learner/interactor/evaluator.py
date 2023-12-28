from typing import Any, Optional, Tuple, Union, Dict

import gymnasium
import gym
import numpy as np

from magicrl.env import BaseVectorEnv, DummyVectorEnv
from magicrl.agents import BaseAgent


class Evaluator:
    def __init__(self, env: Union[gymnasium.Env, BaseVectorEnv], agent: BaseAgent) -> None:
        if isinstance(env, BaseVectorEnv):
            self.env = env
        else:
            self.env = DummyVectorEnv([env])
        self.agent = agent
        self.env_num = self.env.env_num

    def evaluate(self) -> Dict:
        episode_rewards = np.zeros(self.env_num, dtype=np.float32)
        episode_lengths = np.zeros(self.env_num, dtype=np.int32)
        episode_dones = np.zeros(self.env_num, dtype=bool)

        obs, _ = self.env.reset()

        while True:
            act = self.agent.select_action(obs, eval=True)
            obs, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)

            episode_rewards += reward * (1. - episode_dones)
            episode_lengths += 1 - episode_dones
            episode_dones = done

            if np.all(episode_dones):
                break

        avg_reward = np.mean(episode_rewards).item()
        max_reward = np.max(episode_rewards).item()
        min_reward = np.min(episode_rewards).item()
        avg_length = np.mean(episode_lengths).item()
    
        evaluate_summaries = {'avg_reward': avg_reward, 
                              'max_reward': max_reward,
                              'min_reward': min_reward,
                              'avg_length': avg_length}
        
        print(f'Evaluate at step {self.agent.train_step} in {self.env_num} envs: ', 
              f'{np.round(episode_rewards, 1)} ',
              f'avergae length: {avg_length:.1f}, average reward: {avg_reward:.1f}')

        return evaluate_summaries
        



            

        