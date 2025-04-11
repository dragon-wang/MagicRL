import gym


class GymToGymnasium(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}
    
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        # The 'truncated' from gym must be False, 
        # otherwise the 'terminated' transition 
        # cannot be sampled in the reply buffer, 
        # resulting in the inability to learn the final state reward.
        return next_obs, reward, done, False, info
        