import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gymnasium as gym

from magicrl.agents.modelfree import TD3Agent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn import SimpleActor, SimpleCritic
from magicrl.env.maker import make_gym_env


if __name__ == '__main__':

    seed = 10
    train_env_num = 10
    eval_env_num = 5
    learn_id = "test/td3_hopper-"
    env_name = "Hopper-v4"

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, train_envs, eval_envs = make_gym_env(env_name=env_name,
                                              train_env_num=train_env_num,
                                              eval_env_num=eval_env_num,
                                              seed=seed,
                                              dummy=False)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    actor = SimpleActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[400, 300])
    critic1 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])
    critic2 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])


    agent = TD3Agent(actor=actor, critic1=critic1, critic2=critic2, device='cuda')

    # replaybuffer = ReplayBuffer(buffer_size=1000000)
    replaybuffer = VectorBuffer(buffer_size=1000000, buffer_num=train_env_num, buffer_class=ReplayBuffer)

    learner = OffPolicyLearner(explore_step=1000,
                            learn_id=learn_id,
                            train_env=train_envs,
                            eval_env=eval_envs,
                            agent=agent,
                            buffer=replaybuffer,
                            batch_size=100,
                            max_train_step=1000000,
                            learner_log_freq=1000,
                            agent_log_freq=100000,
                            eval_freq=500,
                            resume=False)

    learner.learn()

    # infer_env = gym.make("Hopper-v4", render_mode='human')
    # inferrer = Inferrer(env=infer_env, agent=agent, learn_id=learn_id)
    # inferrer.infer()