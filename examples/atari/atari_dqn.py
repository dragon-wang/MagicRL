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
from magicrl.nn.feature import AtariCNN
from magicrl.env.maker import make_atari_env
from magicrl.env.wrapper import wrap_deepmind

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')  # BreakoutNoFrameskip-v4
    parser.add_argument('--train_num', type=int, default=2)
    parser.add_argument('--eval_num', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--explore_step', type=int, default=10000)
    parser.add_argument('--max_train_step', type=int, default=2000000)  # 10M for BreakoutNoFrameskip-v4
    parser.add_argument('--learn_id', type=str, default='dqn_CartPole')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    parser.add_argument('--scale_obs', action='store_true', default=False)
    parser.add_argument('--collect_per_step', type=int, default=10)


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if not args.infer:
        # make env for train
        env, train_envs, eval_envs = make_atari_env(env_name=args.env,
                                                    train_env_num=args.train_num,
                                                    eval_env_num=args.eval_num,
                                                    seed=args.seed,
                                                    dummy=False,
                                                    scale=args.scale_obs)
    else:
        # make env for infer
        env = wrap_deepmind(args.env, episode_life=False, clip_rewards=False, render_mode='human')

    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n

    feature_net = AtariCNN(num_frames_stack=4, pixel_size=[84, 84])
    q_net = QNet(obs_dim=obs_dim, act_num=act_num, hidden_size=[512, 512], feature_net=feature_net)
    agent = DQNAgent(q_net=q_net,
                     q_lr=1e-4,
                     initial_eps=1.,
                     end_eps=0.05,
                     eps_decay_period=250000,
                     eval_eps=0.005,
                     target_update_freq=500,
                     device=args.device)

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
                                   learner_log_freq=1000,
                                   agent_log_freq=100000,
                                   eval_freq=5000,
                                   resume=args.resume,
                                   collect_per_step=args.collect_per_step)
        learner.learn()

    else:
        inferrer = Inferrer(env=env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)