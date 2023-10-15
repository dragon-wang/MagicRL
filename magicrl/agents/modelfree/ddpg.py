import copy
import torch
import torch.nn.functional as F
import numpy as np
from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class DDPGAgent(BaseAgent):
    """
    Implementation of Deep Deterministic Policy Gradient (DDPG)
    https://arxiv.org/abs/1509.02971
    """
    def __init__(self,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 tau=0.005,  # used to update target network, w' = tau*w + (1-tau)*w'
                 gaussian_noise_sigma=0.1, 
                 **kwargs        
                 ):
        super().__init__(**kwargs)

        self.act_dim = actor.act_dim
        self.act_bound = actor.act_bound

        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau
        self.gaussian_noise_sigma = gaussian_noise_sigma

        self.attr_names.extend(["actor", "target_actor", "critic", "target_critic", "actor_optim", "critic_optim"])


    def select_action(self, obs, eval=False):
        obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)

        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().flatten()
        if eval:
            return action
        else:
            noise = np.random.normal(0, self.gaussian_noise_sigma, size=self.act_dim)
            return (action + noise).clip(-self.act_bound, self.act_bound)

    def train(self, batch):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Compute target Q value
        with torch.no_grad():
            next_act = self.target_actor(next_obs)
            next_Q = self.target_critic(next_obs, next_act).squeeze(1)
            target_Q = rews + (1. - done) * self.gamma * next_Q

        # Compute current Q
        current_Q = self.critic(obs, acts).squeeze(1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        # Optimize actor net
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Optimize critic net
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        soft_target_update(self.actor, self.target_actor, tau=self.tau)
        soft_target_update(self.critic, self.target_critic, tau=self.tau)

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item()}
        return train_summaries
