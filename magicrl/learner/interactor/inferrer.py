from typing import Any, Optional, Tuple, Union, Dict

import gymnasium as gym
import numpy as np

from magicrl.agents import BaseAgent
from magicrl.utils.logger import AgentLogger


class Inferrer:
    def __init__(self, env: gym.Env, agent: BaseAgent, learn_id: str) -> None:
        self.env = env
        self.agent = agent
        self.learn_id = learn_id


    def infer(self, episode_num=5, checkpoint_step=-1) -> None:
        agent_logger = AgentLogger(self.learn_id ,True)
        agent_logger.load_agent(self.agent, checkpoint_step)

        total_reward = 0
        total_length = 0

        for i in range(episode_num):
            episode_reward = 0
            episode_length = 0
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.select_action(np.array([obs]), eval=True)
                obs, reward, terminated, truncated, _ = self.env.step(action.squeeze(0))
                done = np.logical_or(terminated, truncated)
                episode_reward += reward
                episode_length += 1
                if done:
                    total_reward += episode_reward
                    total_length += episode_length
                    print("[episode_id:{}], length:{}, reward:{:.2f}]".format(i+1, episode_length, episode_reward))

        avg_reward = total_reward / episode_num
        avg_length = total_length / episode_num

        print("[infer_result]: average_length: {}, average_reward: {:.2f}".format(avg_length, avg_reward))

        



            

        