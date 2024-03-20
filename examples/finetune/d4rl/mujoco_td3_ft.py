import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import gym
import d4rl
import torch
import numpy as np

from magicrl.agents.modelfree.td3 import TD3Agent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner import Off2OnLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import SimpleActor, SimpleCritic
from magicrl.utils.data_tools import get_d4rl_dataset
from magicrl.env.wrapper.common import GymToGymnasium
from magicrl.env.maker import make_d4rl_env, get_d4rl_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-v2')
    parser.add_argument('--eval_num', type=int, default=10)
    parser.add_argument('--offline_id', type=str, default='td3bc/hop-m-v2')
    parser.add_argument('--offline_step', type=int, default=500000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--explore_step', type=int, default=25000)
    parser.add_argument('--max_train_step', type=int, default=1000000)
    parser.add_argument('--learn_id', type=str, default='finetune/hop-m-v2/ft/td3/test')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    parser.add_argument('--buffer_type', type=int, default=1)
    parser.add_argument('--no_optim', action='store_true', default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_d4rl_space(args.env)

    train_env = make_d4rl_env(env_name=args.env,
                              env_num=1,
                              seed=args.seed,
                              dummy=True)
    
    # 1.Make environment.
    if not args.infer:
        eval_envs = make_d4rl_env(env_name=args.env,
                                  env_num=args.eval_num,
                                  seed=args.seed,
                                  dummy=False)

    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]
    act_bound = action_space.high[0]
 
    actor = SimpleActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[256, 256])
    critic1 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256])
    critic2 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256])

    agent = TD3Agent(actor=actor, 
                     critic1=critic1,
                     critic2=critic2, 
                     actor_lr=3e-4,
                     critic_lr=3e-4,
                     tau=0.005,
                     exploration_noise=0.1,
                     policy_noise=0.2,
                     policy_noise_clip=0.5,
                     policy_delay=2,
                     device=args.device)
    
    attr_names=['train_step','actor', 'target_actor', 
                'critic1', 'target_critic1', 
                'critic2', 'target_critic2']
    
    if not args.no_optim:
        attr_names.extend(['actor_optim', 'critic_optim1', 'critic_optim2'])
    
    # 3.Make Learner and Inferrer.
    # buffer_type=1: Initialize an empty ReplayBuffer for online finetune.
    # buffer_type=2: Initialize an ReplayBuffer with offline data for online finetune. (The offline data will be replaced.)
    # buffer_type=3: Initialize an ReplayBuffer with offline data for online finetune. (The offline data will not be replaced.)
    if not args.infer:
        finetunebuffer = ReplayBuffer(buffer_size=args.buffer_size)
        if args.buffer_type == 2:
            finetunebuffer.init_offline(*get_d4rl_dataset(gym.make(args.env)), args.buffer_size)
        elif args.buffer_type == 3:
            finetunebuffer.init_offline(*get_d4rl_dataset(gym.make(args.env)), args.buffer_size+500000)
        
        learner = Off2OnLearner(offline_id=args.offline_id,
                                offline_step=args.offline_step,
                                attr_names=attr_names,
                                explore_step=args.explore_step,
                                batch_size=args.batch_size,
                                learn_id=args.learn_id,
                                train_env=train_env,
                                eval_env=eval_envs,
                                agent=agent,
                                buffer=finetunebuffer,
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