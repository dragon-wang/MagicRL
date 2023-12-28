from magicrl.env.wrapper import wrap_deepmind
from magicrl.env import SubprocVectorEnv, DummyVectorEnv


def make_atari_env(env_name, train_env_num, eval_env_num, seed, dummy=False, **kwargs):

    VectorEnv = DummyVectorEnv if dummy else SubprocVectorEnv

    train_envs = VectorEnv([wrap_deepmind(env_name, episode_life=True, clip_rewards=True, **kwargs) 
                            for _ in range(train_env_num)])
    
    eval_envs = VectorEnv([wrap_deepmind(env_name, episode_life=False, clip_rewards=False, **kwargs) 
                           for _ in range(eval_env_num)])
    
    train_envs.seed(seed)
    eval_envs.seed(seed)

    return train_envs, eval_envs

def get_atari_space(env_name):
    env = wrap_deepmind(env_name=env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space