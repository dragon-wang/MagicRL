import gym

from magicrl.env import SubprocVectorEnv, DummyVectorEnv
from magicrl.env.wrapper.common import GymToGymnasium


def make_d4rl_env(env_name, eval_env_num, seed, dummy=False):

    VectorEnv = DummyVectorEnv if dummy else SubprocVectorEnv

    eval_envs = VectorEnv([GymToGymnasium(gym.make(env_name)) for _ in range(eval_env_num)])

    eval_envs.seed(seed)

    return eval_envs


def get_d4rl_space(env_name):
    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space