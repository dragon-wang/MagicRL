import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import gym
import d4rl
import torch
import numpy as np

from magicrl.agents.modelfree.sac import SACAgent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import RepapamGaussionActor, SimpleCritic
from magicrl.env.wrapper.common import GymToGymnasium
from magicrl.env.maker import make_d4rl_env, get_d4rl_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-v2')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--eval_num', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--explore_step', type=int, default=25000)
    parser.add_argument('--max_train_step', type=int, default=1000000)
    parser.add_argument('--learn_id', type=str, default='sac/hop-m-v2')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_d4rl_space(args.env)

    train_envs = make_d4rl_env(env_name=args.env,
                              env_num=args.train_num,
                              seed=args.seed,
                              dummy=True)
    
    # 1.Make environment.
    if not args.infer:
        eval_envs = make_d4rl_env(env_name=args.env,
                                  env_num=args.eval_num,
                                  seed=args.seed,
                                  dummy=True)
        
    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]
    act_bound = action_space.high[0]

    actor = RepapamGaussionActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[128, 128])
    critic1 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[128, 128])
    critic2 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[128, 128])

    agent = SACAgent(actor=actor, 
                     critic1=critic1,
                     critic2=critic2, 
                     actor_lr=1e-3,
                     critic_lr=1e-3,
                     alpha_lr=3e-4,
                     tau=0.005,
                     alpha=0.2,
                     auto_alpha=True,
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
                                   learner_log_freq=1000,
                                   agent_log_freq=100000,
                                   eval_freq=5000,
                                   resume=args.resume)
        learner.learn()

    else:
        infer_env = GymToGymnasium(gym.make(args.env, render_mode='human'))
        inferrer = Inferrer(env=infer_env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)