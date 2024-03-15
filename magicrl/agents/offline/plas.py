import copy
import torch
import torch.nn.functional as F

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class PLASAgent(BaseAgent):
    """
    Implementation of Policy in the Latent Action Space(PLAS) in continuous action space
    https://arxiv.org/abs/2011.07213
    """
    def __init__(self,
                 actor: torch.nn.Module,
                 critic1: torch.nn.Module,
                 critic2: torch.nn.Module,
                 cvae: torch.nn.Module,  # generation model
                 critic_lr=1e-3,
                 actor_lr=1e-4,
                 cvae_lr=1e-4,
                 tau=0.005,
                 lmbda=0.75,  # used for double clipped double q-learning
                 max_cvae_iterations=500000,  # the num of iterations when training CVAE model
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.cvae = cvae.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.cvae_optim = torch.optim.Adam(self.cvae.parameters(), lr=cvae_lr)

        self.tau = tau
        self.lmbda = lmbda
        self.max_cvae_iterations = max_cvae_iterations
        self.cvae_iterations= 0

        self.attr_names.extend(['actor', 'critic1', 'critic2', 'cvae', 
                                'target_actor', 'target_critic1', 'target_critic2',
                                'actor_optim', 'critic_optim1', 'critic_optim2', 'cvae_optim',
                                'cvae_iterations'])
        
    def select_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = self.actor(obs, self.cvae.decode)
        return action.cpu().numpy()

    def train_cvae(self, batch):
        """
        Train CVAE one step
        """
        obs, acts = batch["obs"], batch["act"]

        recon_action, mu, log_std = self.cvae(obs, acts)
        cvae_loss = self.cvae.loss_function(recon_action, acts, mu, log_std)

        self.cvae_optim.zero_grad()
        cvae_loss.backward()
        self.cvae_optim.step()

        self.cvae_iterations += 1

        cvae_summaries = {"cvae_loss": cvae_loss.cpu().item()}

        return cvae_summaries

    def train(self, batch):
        # Sample
        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        """
        Train Critic
        """
        with torch.no_grad():
            decode_action_next = self.target_actor(next_obs, self.cvae.decode)

            target_q1 = self.target_critic1(next_obs, decode_action_next)
            target_q2 = self.target_critic2(next_obs, decode_action_next)

            target_q = (self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)).squeeze(1)
            target_q = rews + self.gamma * (1. - done) * target_q

        current_q1 = self.critic1(obs, acts).squeeze(1)
        current_q2 = self.critic2(obs, acts).squeeze(1)

        critic_loss1 = F.mse_loss(current_q1, target_q)
        critic_loss2 = F.mse_loss(current_q2, target_q)

        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        """
        Train Actor
        """
        decode_action = self.actor(obs, self.cvae.decode)
        actor_loss = -self.critic1(obs, decode_action).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        """
        Update target networks
        """
        soft_target_update(self.critic1, self.target_critic1, tau=self.tau)
        soft_target_update(self.critic2, self.target_critic2, tau=self.tau)
        soft_target_update(self.actor, self.target_actor, tau=self.tau)

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": ((critic_loss1 + critic_loss2) / 2).cpu().item(),
                           "q_mean": ((current_q1.mean() + current_q2.mean())/2).cpu().item()}

        return train_summaries