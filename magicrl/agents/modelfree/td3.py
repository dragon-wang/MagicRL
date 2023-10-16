import copy
import torch
import torch.nn.functional as F
import numpy as np
from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class TD3Agent(BaseAgent):
    """
    Implementation of Twin Delayed Deep Deterministic policy gradient (TD3)
    https://arxiv.org/abs/1802.09477
    """
    def __init__(self,
                 actor: torch.nn.Module,
                 critic1: torch.nn.Module,
                 critic2: torch.nn.Module,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 tau=0.005,  # used to update target network, w' = tau*w + (1-tau)*w'
                 exploration_noise=0.1,  # Std of Gaussian exploration noise
                 policy_noise=0.2,  # Std of noise added to target policy during critic update
                 policy_noise_clip=0.5,  # Range to clip target policy noise
                 policy_delay=2,  # Frequency of delayed policy updates
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.act_dim = actor.act_dim
        self.act_bound = actor.act_bound

        # the network and optimizers
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.tau = tau
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.policy_delay = policy_delay

        self.attr_names.extend(['actor', 'target_actor', 
                                'critic1', 'target_critic1', 
                                'critic2', 'target_critic2', 
                                'actor_optim', 'critic_optim1', 'critic_optim2'])

    def select_action(self, obs, eval=False):
        obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)

        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().flatten()
        if eval:
            return action
        else:
            noise = np.random.normal(0, self.exploration_noise, size=self.act_dim)
            return (action + noise).clip(-self.act_bound, self.act_bound)

    def train(self, batch):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Target Policy Smoothing. Add clipped noise to next actions when computing target Q.
        with torch.no_grad():
            noise = torch.normal(mean=0, std=self.policy_noise, size=acts.size()).to(self.device)
            noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            next_act = self.target_actor(next_obs) + noise
            next_act = next_act.clamp(-self.act_bound, self.act_bound)

            # Clipped Double Q-Learning. Compute the min of target Q1 and target Q2
            min_target_q = torch.min(self.target_critic1(next_obs, next_act),
                                     self.target_critic2(next_obs, next_act)).squeeze(1)
            y = rews + self.gamma * (1. - done) * min_target_q

        current_q1 = self.critic1(obs, acts).squeeze(1)
        current_q2 = self.critic2(obs, acts).squeeze(1)

        # TD3 Loss
        critic_loss1 = F.mse_loss(current_q1, y)
        critic_loss2 = F.mse_loss(current_q2, y)

        # Optimize critic net
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        if (self.train_step+1) % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            # Optimize actor net
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            soft_target_update(self.actor, self.target_actor, tau=self.tau)
            soft_target_update(self.critic1, self.target_critic1, tau=self.tau)
            soft_target_update(self.critic2, self.target_critic2, tau=self.tau)
        else:
            actor_loss = torch.tensor(0)


        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss_mean": ((critic_loss1 + critic_loss2) / 2).cpu().item()}

        return train_summaries