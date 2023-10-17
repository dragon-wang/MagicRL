import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import gymnasium as gym
from magicrl.agents.modelfree.td3 import TD3Agent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner.learners import OffPolicyLearner
from magicrl.nn.continuous import MLPQsaNet, DDPGMLPActor
from magicrl.env.gym_env import make_gym_env


seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

train_env, eval_env, observation_space, action_space = make_gym_env("Pendulum-v1", seed)
infer_env = gym.make("Pendulum-v1", render_mode='human')

obs_dim = observation_space.shape[0]
act_dim = action_space.shape[0]
act_bound = action_space.high[0]

actor = DDPGMLPActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[400, 300])

critic1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])
critic2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])

agent = TD3Agent(actor=actor, critic1=critic1, critic2=critic2, device='cuda')

replaybuffer = ReplayBuffer(buffer_size=50000, batch_size=64)

learner = OffPolicyLearner(explore_step=500,
                           learn_id="td3_test_cuda",
                           train_env=train_env,
                           eval_env=eval_env,
                           agent=agent,
                           buffer=replaybuffer,
                           max_train_step=100000,
                           learner_log_freq=1000,
                           agent_log_freq=5000,
                           eval_freq=5000,
                           resume=True)

learner.learn()
# 
# learner.inference(infer_env, 10)
