import copy
import numpy as np
import torch
import torch.nn.functional as F

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class IQLAgent(BaseAgent):
    """
    Implementation of implicit Q-learning (IQL)
    https://arxiv.org/abs/2006.04779
    """
    def __init__(self,
                 actor: torch.nn.Module, 
                 critic_q1: torch.nn.Module, 
                 critic_q2: torch.nn.Module,
                 critic_v: torch.nn.Module,
                 actor_lr=3e-4,
                 critic_q_lr=3e-4,
                 critic_v_lr=3e-4,
                 tau=0.005,
                 expectile=0.7,
                 beta=3.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        self.act_dim = actor.act_dim

        # the network and optimizers
        self.actor = actor.to(self.device)
        self.critic_q1 = critic_q1.to(self.device)
        self.critic_q2 = critic_q2.to(self.device)
        self.critic_v = critic_v.to(self.device)
        self.target_critic_q1 = copy.deepcopy(self.critic_q1).to(self.device)
        self.target_critic_q2 = copy.deepcopy(self.critic_q2).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_q1_optim = torch.optim.Adam(self.critic_q1.parameters(), lr=critic_q_lr)
        self.critic_q2_optim = torch.optim.Adam(self.critic_q2.parameters(), lr=critic_q_lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)

        self.tau = tau
        self.expectile = expectile
        self.beta = beta

        self.attr_names.extend(['actor','critic_q1', 'target_critic_q1', 'critic_q2', 'target_critic_q2', 'critic_v',
                                'actor_optim', 'critic_q1_optim', 'critic_q2_optim', 'critic_v_optim'])


    def select_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action, _ = self.actor.sample(obs, deterministic=eval)

        return action.cpu().numpy()

    def train(self, batch):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        """
        Critic_q Loss
        """
        q1 = self.critic_q1(obs, acts).squeeze(1)
        q2 = self.critic_q2(obs, acts).squeeze(1)

        next_v = self.critic_v(next_obs).squeeze(1).detach()
        y = rews + self.gamma * (1. - done) * next_v

        critic_q1_loss = F.mse_loss(q1, y)
        critic_q2_loss = F.mse_loss(q2, y)  # 两个loss是否一样

        """
        Critic_v Loss
        """
        target_q = torch.min(self.target_critic_q1(obs, acts), self.target_critic_q2(obs, acts)).detach()
        v = self.critic_v(obs)
        # expectile regression 
        diff = target_q - v
        # critic_v_loss = (torch.where(diff > 0, self.expectile, 1-self.expectile) * (diff**2)).mean()
        critic_v_loss = ((self.expectile - (diff < 0.0).float()).abs() * (diff**2)).mean()

        """
        Actor Loss
        """
        log_prob, _ = self.actor.get_logprob_entropy(obs, acts)
        adv = diff.detach()
        exp_adv = (self.beta * adv).exp().clamp(max=100.0).squeeze(1)
        actor_loss = (exp_adv * (-log_prob)).mean()
        
        """
        Update networks
        """
        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        soft_target_update(self.critic_q1, self.target_critic_q1, tau=self.tau)
        soft_target_update(self.critic_q2, self.target_critic_q2, tau=self.tau)

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_q_loss": ((critic_q1_loss + critic_q2_loss) / 2).cpu().item(),
                           "critic_v_loss": critic_v_loss.cpu().item(),
                           "q_mean": ((q1.mean() + q2.mean())/2).cpu().item(),
                           "v_mean":v.mean().cpu().item()}

        return train_summaries
