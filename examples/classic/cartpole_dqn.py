import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
import numpy as np
import gymnasium as gym

from magicrl.agents.modelfree.dqn import DQNAgent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.discrete import QNet
from magicrl.env.maker import make_gym_env, get_gym_space

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--eval_num', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--explore_step', type=int, default=1000)
    parser.add_argument('--max_train_step', type=int, default=20000)
    parser.add_argument('--learn_id', type=str, default='dqn_CartPole')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_gym_space(args.env)

    # 1.Make Learner and Inferrer.
    if not args.infer:
        train_envs, eval_envs = make_gym_env(env_name=args.env,
                                             train_env_num=args.train_num,
                                             eval_env_num=args.eval_num,
                                             seed=args.seed,
                                             dummy=False)

    obs_dim = observation_space.shape[0]
    act_num = action_space.n

    q_net = QNet(obs_dim=obs_dim, act_num=act_num, hidden_size=[256, 256])
    agent = DQNAgent(q_net=q_net,
                     q_lr=1e-3,
                     initial_eps=0.1,
                     end_eps=0.001,
                     eps_decay_period=2000,
                     eval_eps=0.001,
                     target_update_freq=10,
                     device=args.device)

    # 3.Make Learner and Inferrer.
    if not args.infer:
        if args.train_num == 1:
            replaybuffer = ReplayBuffer(buffer_size=args.buffer_size)
        else:
            replaybuffer = VectorBuffer(buffer_size=args.buffer_size, 
                                        buffer_num=args.train_num, 
                                        buffer_class=ReplayBuffer)
        
        learner = OffPolicyLearner(explore_step=args.explore_step,
                                   batch_size=args.batch_size,
                                   learn_id=args.learn_id,
                                   train_env=train_envs,
                                   eval_env=eval_envs,
                                   agent=agent,
                                   buffer=replaybuffer,
                                   max_train_step=args.max_train_step,
                                   learner_log_freq=500,
                                   agent_log_freq=5000,
                                   eval_freq=1000,
                                   resume=args.resume)
        learner.learn()

    else:
        infer_env = gym.make(args.env, render_mode='human')
        inferrer = Inferrer(env=infer_env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)