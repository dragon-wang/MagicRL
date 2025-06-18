import numpy as np
import gymnasium
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack


def wrap_deepmind(env_name, episode_life=True, clip_rewards=True, frame_stack=4, scale=False, render_mode=None):


    env = gymnasium.make(env_name, render_mode=render_mode)

    env = AtariPreprocessing(env=env,
                             noop_max=30,
                             frame_skip=4,
                             screen_size=84,
                             terminal_on_life_loss=episode_life,
                             grayscale_obs=True,
                             grayscale_newaxis=False,
                             scale_obs=scale)
    
    env = FrameStack(env=env, num_stack=frame_stack, lz4_compress=False)

    if clip_rewards:
        env = ClipRewardEnv(env=env)

    return env
    
class ClipRewardEnv(gymnasium.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        return np.sign(reward)