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
from magicrl.nn.feature import AtariCNN
from magicrl.env.maker import make_atari_env, get_atari_space
from magicrl.env.wrapper import wrap_deepmind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--train_num', type=int, default=5)
    parser.add_argument('--eval_num', type=int, default=5)
    parser.add_argument('--traj_length', type=int, default=1024)
    parser.add_argument('--max_train_step', type=int, default=2000000)
    parser.add_argument('--learn_id', type=str, default='ppo_atari')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_atari_space(args.env)

    # 1.Make train and evaluation envs.
    if not args.infer:
        train_envs, eval_envs = make_atari_env(env_name=args.env,
                                               train_env_num=args.train_num,
                                               eval_env_num=args.eval_num,
                                               seed=args.seed,
                                               dummy=False,
                                               scale=False)
        
    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    
    act_num = action_space.n

    feature_net = AtariCNN(num_frames_stack=4, pixel_size=[84, 84])
    actor = CategoricalActor(obs_dim=obs_dim, act_num=act_num, 
                             hidden_size=[512, 512],hidden_activation=nn.Tanh,
                             feature_net=feature_net)

        
        
    critic = SimpleCritic(obs_dim=obs_dim, act_dim=0, hidden_size=[512, 512], 
                          hidden_activation=nn.Tanh, feature_net=feature_net)


    agent = PPO_Agent(actor=actor, 
                     critic=critic,
                     actor_lr=1e-3,
                     critic_lr=1e-3,
                     gae_lambda=0.95,
                     gae_normalize=True,
                     clip_pram=0.2,
                     ent_coef=0.01,
                     use_grad_clip=True,
                     use_lr_decay=True,
                     train_actor_iters=10,
                     train_critic_iters=10,
                     max_train_step=args.max_train_step,
                     device=args.device)

    # 3.Make Learner and Inferrer.
    if not args.infer:
        if args.train_num == 1:
            trajectoryBuffer = TrajectoryBuffer(buffer_size=args.traj_length)
        else:
            trajectoryBuffer = VectorBuffer(buffer_size=args.traj_length  * args.train_num,  # total size of n buffer.
                                            buffer_num=args.train_num, 
                                            buffer_class=TrajectoryBuffer)
        
        learner = OnPolicyLearner( trajectory_length=args.traj_length,
                                   learn_id=args.learn_id,
                                   train_env=train_envs,
                                   eval_env=eval_envs,
                                   agent=agent,
                                   buffer=trajectoryBuffer,
                                   max_train_step=args.max_train_step,
                                   learner_log_freq=5000,
                                   agent_log_freq=100000,
                                   eval_freq=5000,
                                   resume=args.resume)
        learner.learn()

    else:
        infer_env = wrap_deepmind(args.env, episode_life=False, clip_rewards=False, render_mode='human')
        inferrer = Inferrer(env=infer_env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)