from typing import Any, Optional, Tuple, Union, Dict

from tqdm import tqdm
import gymnasium as gym
import numpy as np

from magicrl.env import BaseVectorEnv, DummyVectorEnv
from magicrl.data import BaseBuffer
from magicrl.agents import BaseAgent


def _explore_tqdm(step: int, random: bool):
    t = tqdm(range(step))
    if random:
        t.set_description('Explore randomly before train')
    else:
        t.set_description('Explore by loaded agent before train')
    return t


class Collector:
    def __init__(self, 
                 env: Union[gym.Env, BaseVectorEnv],
                 agent: BaseAgent, 
                 buffer: BaseBuffer
                 ) -> None:
        if isinstance(env, gym.Env):
            self.env = DummyVectorEnv([env])
        else:
            self.env = env
        self.agent = agent
        self.buffer = buffer

        self.last_obs, _ = self.env.reset()

    def collect(self, n_step: int = None, is_explore=False, random: bool = False):
        obs = self.last_obs

        t = _explore_tqdm(n_step, random) if is_explore else range(n_step)
        for _ in t:
            if random:
                act = [ac.sample() for ac in self.env.action_spaces]
            else:
                act = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            
            transitions = [{"obs": obs[i], 
                            "act": act[i],
                            "rew": reward[i],
                            "next_obs": next_obs[i],
                            'done': done[i]} 
                            for i in range(self.env.env_num)]

            self.buffer.add(transitions if self.env.env_num > 1 else transitions[0])

            if np.any(done):
                reset_id = np.where(done)[0]
                next_obs[reset_id], _ = self.env.reset(reset_id)        
            obs = next_obs
        self.last_obs = obs
                