import gymnasium as gym

def make_gym_env(env_name, seed):
    env = gym.make(env_name)
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    train_env.action_space.seed(seed=seed)
    eval_env.action_space.seed(seed=seed)

    train_env.reset(seed=seed)
    eval_env.reset(seed=seed)

    return train_env, eval_env, env.observation_space, env.action_space
    