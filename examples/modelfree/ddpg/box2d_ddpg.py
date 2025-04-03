import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import torch
import numpy as np
import gymnasium

from magicrl.agents.modelfree.ddpg import DDPGAgent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import SimpleActor, SimpleCritic
from magicrl.env.maker import make_gymnasium_env, get_gymnasium_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--train_num', type=int, default=5)
    parser.add_argument('--eval_num', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=300000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--explore_step', type=int, default=10000)
    parser.add_argument('--max_train_step', type=int, default=1000000)
    parser.add_argument('--learn_id', type=str, default='ddpg_BipedalWalker')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_gymnasium_space(args.env)

    # 1.Make environments.
    if not args.infer:
        train_envs, eval_envs = make_gymnasium_env(env_name=args.env,
                                                   train_env_num=args.train_num,
                                                   eval_env_num=args.eval_num,
                                                   seed=args.seed,
                                                   dummy=False)
        
    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]
    act_bound = action_space.high[0]

    actor = SimpleActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[256, 256])
    critic = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256])

    agent = DDPGAgent(actor=actor, 
                      critic=critic, 
                      actor_lr=3e-4,
                      critic_lr=3e-4,
                      tau=0.005,
                      exploration_noise=0.1,
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
        infer_env = gymnasium.make(args.env, render_mode='human')
        inferrer = Inferrer(env=infer_env, agent=agent, learn_id=args.learn_id)
        inferrer.infer(episode_num=10, checkpoint_step=args.infer_step)