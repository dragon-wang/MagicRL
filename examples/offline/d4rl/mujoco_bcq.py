import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import gym
import d4rl
import torch
import numpy as np

from magicrl.agents.offline import BCQAgent
from magicrl.data.buffers import ReplayBuffer
from magicrl.learner import OfflineLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import SimpleCritic, CVAE, PerturbActor
from magicrl.utils.data_tools import get_d4rl_dataset
from magicrl.env.wrapper.common import GymToGymnasium
from magicrl.env.maker import make_d4rl_env, get_d4rl_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-v2')
    parser.add_argument('--eval_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_train_step', type=int, default=1000000)
    parser.add_argument('--learn_id', type=str, default='bcq_hopper-medium-v2')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_step', type=int, default=-1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_space, action_space = get_d4rl_space(args.env)
    
    # 1.Make environment.
    if not args.infer:
        eval_envs = make_d4rl_env(env_name=args.env,
                                  eval_env_num=args.eval_num,
                                  seed=args.seed,
                                  dummy=False)

    # 2.Make agent.
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]
    act_bound = action_space.high[0]

    critic1 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])
    critic2 = SimpleCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])
    perturb = PerturbActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,hidden_size=[400, 300], phi=0.05)
    cvae = CVAE(obs_dim=obs_dim, act_dim=act_dim, latent_dim=2*act_dim, act_bound=act_bound)

    agent = BCQAgent(critic1=critic1,
                     critic2=critic2,
                     perturb=perturb,
                     cvae=cvae,  
                     critic_lr=1e-3,
                     per_lr=1e-3,
                     cvae_lr=1e-3,
                     tau=0.005,
                     lmbda=0.75,
                     device=args.device)
    
    # 3.Make Learner and Inferrer.
    if not args.infer:
        offlinebuffer = ReplayBuffer()
        offlinebuffer.init_offline(*get_d4rl_dataset(gym.make(args.env)))

        learner = OfflineLearner(batch_size=args.batch_size,
                                 learn_id=args.learn_id,
                                 eval_env=eval_envs,
                                 agent=agent,
                                 buffer=offlinebuffer,
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


