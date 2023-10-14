from tqdm import tqdm
from magicrl.data.buffers import BaseBuffer
import numpy as np

def explore_randomly(env, buffer: BaseBuffer, explore_step: int):
    obs, _ = env.reset()
    t = tqdm(range(explore_step))
    t.set_description("Explore randomly before train")
    for _ in t:
        act = env.action_space.sample()
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = np.logical_or(terminated, truncated)
        transition = {"obs": obs,
                        "act": act,
                        "rew": rew,
                        "next_obs": next_obs,
                        "done": done}
        buffer.add(transition)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs


def explore_by_agent(env, agent, buffer: BaseBuffer, explore_step: int):
    obs, _ = env.reset()
    t = tqdm(range(explore_step))
    t.set_description("Explore by agent before train")
    for _ in t:
        act = agent.select_action(obs)
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = np.logical_or(terminated, truncated)
        transition = {"obs": obs,
                        "act": act,
                        "rew": rew,
                        "next_obs": next_obs,
                        "done": done}
        buffer.add(transition)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
