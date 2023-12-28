from typing import Any, Optional, Tuple, Union, Dict

from tqdm import tqdm
import gymnasium
import numpy as np

from magicrl.env import BaseVectorEnv, DummyVectorEnv
from magicrl.data import BaseBuffer, VectorBuffer
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
                 env: Union[gymnasium.Env, BaseVectorEnv],
                 agent: BaseAgent, 
                 buffer: Union[BaseBuffer, VectorBuffer]
                 ) -> None:
        if isinstance(env, gymnasium.Env):
            self.env = DummyVectorEnv([env])
        else:
            self.env = env
        self.agent = agent
        self.buffer = buffer

        self.last_obs, _ = self.env.reset()
        self.cur_steps = np.zeros(self.env.env_num, dtype=np.int32)

    def collect(self, n_step: int = None, is_explore=False, random: bool = False, save_next_obs=False):
        obs = self.last_obs

        t = _explore_tqdm(n_step, random) if is_explore else range(n_step)
        for _ in t:
            if random:
                act = [ac.sample() for ac in self.env.action_spaces]
            else:
                act = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            self.cur_steps += 1
            
            if save_next_obs:  
                transitions = [{
                                "obs": obs[i],
                                "act": act[i],
                                "rew": reward[i],
                                'done': done[i],
                                'term': terminated[i],
                                'trun': truncated[i],
                                'next_obs': next_obs[i]
                                } 
                                for i in range(self.env.env_num)]
            else:  
                transitions = [{
                                "obs": obs[i],
                                "act": act[i],
                                "rew": reward[i],
                                'done': done[i],
                                'term': terminated[i],
                                'trun': truncated[i]
                                } 
                                for i in range(self.env.env_num)]
                
            # # The rnn_state is like: {'hidden': np.ndarray, 'cell': np.ndarray}
            # if rnn_state is not None:
            #     for i in range(self.env.env_num):
            #         transitions[i]['rnn_state'] = rnn_state[i]
            
            if self.env.env_num > 1:
                self.buffer.add(transitions, self.cur_steps)
            else:
                self.buffer.add(transitions[0], self.cur_steps[0])

            if np.any(done):
                reset_id = np.where(done)[0]
                next_obs[reset_id], _ = self.env.reset(reset_id)    
                self.cur_steps[reset_id] = 0    
            obs = next_obs

        self.last_obs = obs
                