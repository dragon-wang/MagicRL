import gym


class GymToGymnasium(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}
    
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, done, info
        