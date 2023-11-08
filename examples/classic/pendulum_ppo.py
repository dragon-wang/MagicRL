import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
from torch import nn
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from magicrl.agents.modelfree.ppo import PPO_Agent
from magicrl.data.buffers import TrajectoryBuffer, VectorBuffer
from magicrl.learner import OnPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import GaussionActor, SimpleCritic
from magicrl.nn.discrete import CategoricalActor

from magicrl.env.maker import make_gym_env, get_gym_space

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--eval_num', type=int, default=1)
    parser.add_argument('--traj_length', type=int, default=500)
    parser.add_argument('--max_train_step', type=int, default=500000)
    parser.add_argument('--learn_id', type=str, default='test_ppo')
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
        
    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    

    if isinstance(action_space, Discrete):
        act_num = action_space.n
        actor = CategoricalActor(obs_dim=obs_dim, act_num=act_num, 
                                 hidden_size=[128, 128],hidden_activation=nn.Tanh)
    elif isinstance(action_space, Box):
        act_dim = action_space.shape[0]
        act_bound = action_space.high[0]
        actor = GaussionActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, 
                              hidden_size=[128, 128], hidden_activation=nn.Tanh)
    else:
        raise TypeError
        
        
    critic = SimpleCritic(obs_dim=obs_dim, act_dim=0, hidden_size=[128, 128], hidden_activation=nn.Tanh)


    agent = PPO_Agent(actor=actor, 
                     critic=critic,
                     actor_lr=3e-4,
                     critic_lr=3e-4,
                     gae_lambda=0.95,
                     gae_normalize=False,
                     clip_pram=0.2,
                     train_actor_iters=10,
                     train_critic_iters=10,
                     device=args.device)

    # 3.Make Learner and Inferrer.
    if not args.infer:
        if args.train_num == 1:
            trajectoryBuffer = TrajectoryBuffer(buffer_size=args.traj_length)
        else:
            trajectoryBuffer = VectorBuffer(buffer_size=args.traj_length, 
                                        buffer_num=args.train_num, 
                                        buffer_class=TrajectoryBuffer)
        
        learner = OnPolicyLearner( trajectory_length=args.traj_length,
                                   learn_id=args.learn_id,
                                   train_env=train_envs,
                                   eval_env=eval_envs,
                                   agent=agent,
                                   buffer=trajectoryBuffer,
                                   max_train_step=args.max_train_step,
                                   learner_log_freq=2000,
                                   agent_log_freq=5000,
                                   eval_freq=2000,
                                   resume=args.resume)
        learner.learn()

    else:
        infer_env = gym.make(args.env, render_mode='human')
        inferrer = Inferrer(env=infer_env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)