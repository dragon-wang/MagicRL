import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import gymnasium as gym
from magicrl.agents.modelfree.td3 import TD3Agent
from magicrl.data.buffers import ReplayBuffer, VectorBuffer
from magicrl.learner import OffPolicyLearner
from magicrl.learner.interactor import Inferrer
from magicrl.nn.continuous import MLPQsaNet, DDPGMLPActor
from magicrl.env.gym_env import make_gym_env
from magicrl.env import SubprocVectorEnv, DummyVectorEnv


if __name__ == '__main__':
    seed = 10
    env_num = 10

    torch.manual_seed(seed)
    np.random.seed(seed)


    # train_env, eval_env, observation_space, action_space = make_gym_env("Hopper-v4", seed)
    # infer_env = gym.make("Hopper-v4", render_mode='human')
    train_envs = SubprocVectorEnv([gym.make("Hopper-v4", render_mode = None) for i in range(env_num)])
    env = gym.make("Hopper-v4")
    eval_envs = SubprocVectorEnv([gym.make("Hopper-v4", render_mode = None) for i in range(env_num)])

    infer_env = gym.make("Hopper-v4", render_mode='human')

    learn_id = "test/td3_hopper-v4_venvnew1"

    train_envs.seed(10)
    eval_envs.seed(10)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    actor = DDPGMLPActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, hidden_size=[400, 300])

    critic1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])
    critic2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300])

    agent = TD3Agent(actor=actor, critic1=critic1, critic2=critic2, device='cuda')

    # replaybuffer = ReplayBuffer(buffer_size=1000000)
    replaybuffer = VectorBuffer(buffer_size=1000000, buffer_num=env_num, buffer_class=ReplayBuffer)

    learner = OffPolicyLearner(explore_step=10000,
                            learn_id=learn_id,
                            train_env=train_envs,
                            eval_env=eval_envs,
                            agent=agent,
                            buffer=replaybuffer,
                            batch_size=100,
                            max_train_step=2000000,
                            learner_log_freq=1000,
                            agent_log_freq=5000,
                            eval_freq=1000,
                            resume=False)

    learner.learn()

    # inferrer = Inferrer(env=infer_env, agent=agent, learn_id=learn_id)
    # inferrer.infer()