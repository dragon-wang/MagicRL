import gymnasium as gym
from magicrl.env import SubprocVectorEnv, DummyVectorEnv


def make_gym_env(env_name, train_env_num, eval_env_num, seed, dummy=False):
    
    env = gym.make(env_name)
    
    VectorEnv = DummyVectorEnv if dummy else SubprocVectorEnv

    train_envs = VectorEnv([gym.make(env_name) for _ in range(train_env_num)])
    eval_envs = VectorEnv([gym.make(env_name) for _ in range(eval_env_num)])
    
    train_envs.seed(seed)
    eval_envs.seed(seed)

    return env, train_envs, eval_envs
    