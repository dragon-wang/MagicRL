import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from magicrl.agents.modelfree.dqn import DQNAgent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner.learners import OffPolicyLearner
from magicrl.nn.common import MLP
import gymnasium as gym


train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
infer_env = gym.make("CartPole-v1", render_mode='human')

obs_dim = train_env.observation_space.shape[0]
act_dim = train_env.action_space.n

q_net = MLP(input_dim=obs_dim, output_dim=act_dim, hidden_size=[256, 256])

agent = DQNAgent(Q_net=q_net, action_dim=act_dim, device='cpu')

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

# learner.learn()
# 
learner.inference(infer_env, 10)
