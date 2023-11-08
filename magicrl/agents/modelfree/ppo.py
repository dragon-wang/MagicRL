import copy
import torch
import torch.nn.functional as F
import numpy as np

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class PPO_Agent(BaseAgent):
    """
    Implementation of Proximal Policy Optimization (PPO)
    This is the version of "PPO-Clip"
    https://arxiv.org/abs/1707.06347
    """
    def __init__(self,
                 actor: torch.nn.Module,  # actor
                 critic: torch.nn.Module,  # critic
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gae_lambda=0.95,
                 gae_normalize=False,
                 clip_pram=0.2,
                 train_actor_iters=10,
                 train_critic_iters=10,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # the network and optimizer
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gae_lambda = gae_lambda
        self.gae_normalize = gae_normalize
        self.clip_pram = clip_pram
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        self.attr_names.extend(['actor','critic', 'actor_optim', 'critic_optim'])

    def select_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action, eval_action = self.actor(obs)

            if eval:
                action = eval_action 

        return action.cpu().numpy()

    def _compute_gae(self, values, rews, next_values, done):
        gae = 0
        gae_advs = torch.zeros_like(rews)
        for i in reversed(range(len(rews))):
            delta = rews[i] + self.gamma * next_values[i] * (1 - done[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - done[i])
            gae_advs[i] = gae

        target_values = gae_advs + values

        if self.gae_normalize:
            gae_advs = (gae_advs - torch.mean(gae_advs) / torch.std(gae_advs))
        
        return gae_advs, target_values
        

    def train(self, batch):
        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']
        
        with torch.no_grad():
            values = self.critic(obs).squeeze()
            next_values = self.critic(next_obs).squeeze()
            old_log_probs = self.actor.get_log_prob(obs, acts)

        gae_advs, target_values = self._compute_gae(values, rews, next_values, done)
            
        # Train policy with multiple steps of gradient descent
        for _ in range(self.train_actor_iters):
            new_log_probs = self.actor.get_log_prob(obs, acts)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surrogate = ratios * gae_advs
            clipped_surrogate = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * gae_advs

            actor_loss = -(torch.min(surrogate, clipped_surrogate)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # Train value function with multiple steps of gradient descent
        for _ in range(self.train_critic_iters):
            values = self.critic(obs).squeeze()
            critic_loss = F.mse_loss(target_values, values)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item()}

        return train_summaries 
