import copy
import torch
import torch.nn.functional as F
import numpy as np
from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class SACAgent(BaseAgent):
    """
    Implementation of Soft Actor-Critic (SAC)
    https://arxiv.org/abs/1812.05905(SAC 2019)
    """
    def __init__(self,
                 actor: torch.nn.Module,  # actor
                 critic1: torch.nn.Module,  # critic
                 critic2: torch.nn.Module,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=0.05,
                 alpha=0.5,
                 auto_alpha=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.act_dim = actor.act_dim

        # the network and optimizer
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha

        self.attr_names.extend(['actor','critic1', 'target_critic1', 'critic2', 'target_critic2', 
                                'actor_optim', 'critic_optim1', 'critic_optim2'])

        if self.auto_alpha:
            self.target_entropy = -self.act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=actor_lr)
            self.alpha = torch.exp(self.log_alpha)
            self.attr_names.extend(['log_alpha', 'alpha_optim'])

    def select_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
            action, log_prob, mu_action = self.actor(obs)

            if eval:
                action = mu_action  # if eval, use mu as the action

        return action.cpu().numpy().flatten()

    def train(self, batch):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # compute actor Loss
        a, log_prob, _ = self.actor(obs)
        min_q = torch.min(self.critic1(obs, a), self.critic2(obs, a)).squeeze(1)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        # compute critic Loss
        q1 = self.critic1(obs, acts).squeeze(1)
        q2 = self.critic2(obs, acts).squeeze(1)
        with torch.no_grad():
            next_a, next_log_prob, _ = self.actor(next_obs)
            min_target_next_q = torch.min(self.target_critic1(next_obs, next_a), self.target_critic2(next_obs, next_a)).squeeze(1)
            y = rews + self.gamma * (1. - done) * (min_target_next_q - self.alpha * next_log_prob)

        critic_loss1 = F.mse_loss(q1, y)
        critic_loss2 = F.mse_loss(q2, y)

        # Update actor network parameter
        # actor network's update should be done before updating critic network, or there will make some errors.
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update critic network1 parameter
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        # Update critic network2 parameter
        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0)


        soft_target_update(self.critic1, self.target_critic1, tau=self.tau)
        soft_target_update(self.critic2, self.target_critic2, tau=self.tau)

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss_mean": ((critic_loss1 + critic_loss2) / 2).cpu().item(),
                           "alpha_loss": alpha_loss.cpu().item()}

        return train_summaries

