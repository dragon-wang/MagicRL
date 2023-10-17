import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import gymnasium as gym
from magicrl.agents.modelfree.dqn import DQNAgent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner.learners import OffPolicyLearner
from magicrl.nn.discrete import MLPQsNet
from magicrl.env.gym_env import make_gym_env
from magicrl.env.gym_env import make_gym_env


seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

train_env, eval_env, observation_space, action_space = make_gym_env("CartPole-v1", seed)
infer_env = gym.make("CartPole-v1", render_mode='human')

obs_dim = observation_space.shape[0]
act_num = action_space.n

q_net = MLPQsNet(obs_dim=obs_dim, act_num=act_num, hidden_size=[256, 256])

agent = DQNAgent(q_net=q_net, device='cpu')

replaybuffer = ReplayBuffer(buffer_size=5000, batch_size=128)

learner = OffPolicyLearner(explore_step=500,
                           learn_id="dqn_test",
                           train_env=train_env,
                           eval_env=eval_env,
                           agent=agent,
                           buffer=replaybuffer,
                           max_train_step=10000,
                           learner_log_freq=500,
                           agent_log_freq=1000,
                           eval_freq=1000,
                           resume=False)

learner.learn()
# 
# learner.inference(infer_env, 10)
