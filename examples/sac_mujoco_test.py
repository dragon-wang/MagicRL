import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import gymnasium as gym
from magicrl.agents.modelfree.sac import SACAgent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner.learners import OffPolicyLearner
from magicrl.nn.continuous import MLPQsaNet, MLPSquashedReparamGaussianPolicy
from magicrl.env.gym_env import make_gym_env


seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

train_env, eval_env, observation_space, action_space = make_gym_env("Hopper-v4", seed)
infer_env = gym.make("Hopper-v4", render_mode='human')

obs_dim = observation_space.shape[0]
act_dim = action_space.shape[0]
act_bound = action_space.high[0]

actor = MLPSquashedReparamGaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[256, 256])

critic1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256])
critic2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256])

agent = SACAgent(actor=actor, critic1=critic1, critic2=critic2, auto_alpha=True, device='cuda')

replaybuffer = ReplayBuffer(buffer_size=1000000, batch_size=256)

learner = OffPolicyLearner(explore_step=10000,
                           learn_id="sac_hopperh-v4_test",
                           train_env=train_env,
                           eval_env=eval_env,
                           agent=agent,
                           buffer=replaybuffer,
                           max_train_step=2000000,
                           learner_log_freq=1000,
                           agent_log_freq=5000,
                           eval_freq=5000,
                           resume=False)

learner.learn()
# 
# learner.inference(infer_env, 10)
