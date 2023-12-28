import gymnasium

from magicrl.env import SubprocVectorEnv, DummyVectorEnv


def make_gymnasium_env(env_name, train_env_num, eval_env_num, seed, dummy=False):
    
    VectorEnv = DummyVectorEnv if dummy else SubprocVectorEnv

    train_envs = VectorEnv([gymnasium.make(env_name) for _ in range(train_env_num)])
    eval_envs = VectorEnv([gymnasium.make(env_name) for _ in range(eval_env_num)])
    
    train_envs.seed(seed)
    eval_envs.seed(seed)

    return train_envs, eval_envs


def get_gymnasium_space(env_name):
    env = gymnasium.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space