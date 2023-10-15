import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from magicrl.agents.modelfree.dqn import DQNAgent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner.learners import OffPolicyLearner
from magicrl.nn.discrete import MLPQsNet
import gymnasium as gym


train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
infer_env = gym.make("CartPole-v1", render_mode='human')

obs_dim = train_env.observation_space.shape[0]
act_num = train_env.action_space.n

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
                           resume=True)

learner.learn()
# 
# learner.inference(infer_env, 10)
