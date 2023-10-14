import numpy as np
from magicrl.agents.base import BaseAgent


def evaluate_agent(eval_env, agent: BaseAgent, episode_num: int, render=False):

    total_reward = 0
    total_length = 0

    for i in range(episode_num):
        episode_reward = 0
        episode_length = 0
        obs, _ = eval_env.reset()
        done = False
        while not done:
            if render:
                eval_env.render()
            action = agent.select_action(obs, eval=True)
            # action = action[0] if isinstance(action, tuple) else action
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = np.logical_or(terminated, truncated)
            episode_reward += reward
            episode_length += 1
            if done:
                total_reward += episode_reward
                total_length += episode_length

    avg_reward = total_reward / episode_num
    avg_length = total_length / episode_num

    evaluate_summaries = {"eval_episode_length": avg_length, "eval_episode_reward": avg_reward}
    return evaluate_summaries