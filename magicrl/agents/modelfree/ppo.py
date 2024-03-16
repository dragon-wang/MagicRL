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
                 ent_coef=0.01,
                 use_grad_clip=False,
                 use_lr_decay=False,
                 train_actor_iters=10,
                 train_critic_iters=10,
                 max_train_step=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # the network and optimizer
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gae_lambda = gae_lambda
        self.gae_normalize = gae_normalize
        self.clip_pram = clip_pram
        self.ent_coef = ent_coef
        self.use_grad_clip = use_grad_clip
        self.use_lr_decay  = use_lr_decay
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.max_train_step = max_train_step

        self.attr_names.extend(['actor','critic', 'actor_optim', 'critic_optim'])

    def select_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action, _ = self.actor.sample(obs, deterministic=eval)

        return action.cpu().numpy()
    
    def _lr_decay(self):
        actor_lr_now = self.actor_lr * (1 - self.train_step / self.max_train_step)
        critic_lr_now = self.critic_lr * (1 - self.train_step / self.max_train_step)
        for p in self.actor_optim.param_groups:
            p['lr'] = actor_lr_now
        for p in self.critic_optim.param_groups:
            p['lr'] = critic_lr_now

    def train(self, batch):
        obs, acts, values, old_log_probs, gae_advs = batch['obs'], batch['act'], batch['values'], batch['log_probs'], batch['gae_advs']
        target_values = gae_advs + values
        
        # Train policy with multiple steps of gradient descent
        for _ in range(self.train_actor_iters):
            new_log_probs, new_entropy = self.actor.get_logprob_entropy(obs, acts)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surrogate = ratios * gae_advs
            clipped_surrogate = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * gae_advs

            entropy_loss = new_entropy.mean()
            actor_loss = -(torch.min(surrogate, clipped_surrogate)).mean() - self.ent_coef * entropy_loss
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

        # Train value function with multiple steps of gradient descent
        for _ in range(self.train_critic_iters):
            values = self.critic(obs).squeeze()
            critic_loss = F.mse_loss(target_values, values)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

        if self.use_lr_decay:
            self._lr_decay()

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item(),
                           "entropy": entropy_loss.cpu().item(),
                           'v_mean': values.mean().cpu().item()}

        return train_summaries 
