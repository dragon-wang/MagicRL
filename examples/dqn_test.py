import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gymnasium as gym

from magicrl.agents.modelfree import DQNAgent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn import QNet
from magicrl.env.maker import make_gym_env


if __name__ == '__main__':

    seed = 10
    train_env_num = 10
    eval_env_num = 5
    learn_id = "test/dqn_cartpole-v4"
    env_name = "CartPole-v1"

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, train_envs, eval_envs = make_gym_env(env_name=env_name,
                                              train_env_num=train_env_num,
                                              eval_env_num=eval_env_num,
                                              seed=seed,
                                              dummy=False)


    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n

    q_net = QNet(obs_dim=obs_dim, act_num=act_num, hidden_size=[256, 256])
    agent = DQNAgent(q_net=q_net, device='cpu')

    # replaybuffer = ReplayBuffer(buffer_size=5000)
    replaybuffer = VectorBuffer(buffer_size=5000, buffer_num=train_env_num, buffer_class=ReplayBuffer)

    learner = OffPolicyLearner(explore_step=500,
                            learn_id=learn_id,
                            train_env=train_envs,
                            eval_env=eval_envs,
                            agent=agent,
                            buffer=replaybuffer,
                            batch_size=128,
                            max_train_step=10000,
                            learner_log_freq=500,
                            agent_log_freq=1000,
                            eval_freq=1000,
                            resume=False)

    learner.learn()
    # 
    # infer_env = gym.make("CartPole-v1", render_mode='human')
    # inferrer = Inferrer(env=infer_env, agent=agent, learn_id=learn_id)
    # inferrer.infer()
