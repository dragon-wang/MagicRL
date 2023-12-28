import copy
import torch
import torch.nn.functional as F

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class BCQAgent(BaseAgent):
    """
    Implementation of Batch-Constrained deep Q-learning(BCQ) in continuous action space
    https://arxiv.org/abs/1812.02900
    """
    def __init__(self,
                 critic1: torch.nn.Module,
                 critic2: torch.nn.Module,
                 perturb: torch.nn.Module,  # perturbation model
                 cvae: torch.nn.Module,  # generation model
                 critic_lr=1e-3,
                 per_lr=1e-3,
                 cvae_lr=1e-3,
                 tau=0.005,
                 lmbda=0.75,  # used for double clipped double q-learning
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.perturb = perturb.to(self.device)
        self.cvae = cvae.to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.target_perturb = copy.deepcopy(self.perturb).to(self.device)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.perturb_optim = torch.optim.Adam(self.perturb.parameters(), lr=per_lr)
        self.cvae_optim = torch.optim.Adam(self.cvae.parameters(), lr=cvae_lr)

        self.tau = tau
        self.lmbda = lmbda

        self.attr_names.extend(['critic1', 'critic2', 'perturb', 'cvae', 
                                'target_critic1', 'target_critic2', 'target_perturb'
                                'critic_optim1', 'critic_optim2', 'perturb_optim', 'cvae_optim'])

    def select_action(self, obs, eval=True):
        with torch.no_grad():
            obs_len = obs.shape[0]
            obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.repeat_interleave(obs, repeats=100, dim=0)

            generated_action = self.cvae.decode(obs, z_device=self.device)
            perturbed_action = self.perturb(obs, generated_action)
            q1 = self.critic1(obs, perturbed_action)
            
            ind = q1.reshape(obs_len, 100, 1).argmax(1).squeeze().cpu()
            ind = ind + torch.arange(obs_len) * 100
            action = perturbed_action[ind]
        return action.cpu().numpy()

    def train(self, batch):
        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        """
        CVAE Loss (the generation model)
        """
        recon_action, mu, log_std = self.cvae(obs, acts)
        cvae_loss = self.cvae.loss_function(recon_action, acts, mu, log_std)

        self.cvae_optim.zero_grad()
        cvae_loss.backward()
        self.cvae_optim.step()

        """
        Critic Loss
        """
        with torch.no_grad():
            # generate 10 actions for every next_obs
            next_obs = torch.repeat_interleave(next_obs, repeats=10, dim=0).to(self.device)
            generated_action = self.cvae.decode(next_obs, z_device=self.device)
            # perturb the generated action
            perturbed_action = self.target_perturb(next_obs, generated_action)
            # compute target Q value of perturbed action
            target_q1 = self.target_critic1(next_obs, perturbed_action)
            target_q2 = self.target_critic2(next_obs, perturbed_action)
            # soft clipped double q-learning
            target_q = self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)
            # take max over each action sampled from the generation and perturbation model
            target_q = target_q.reshape(obs.shape[0], 10, 1).max(1)[0].squeeze(1)
            target_q = rews + self.gamma * (1. - done) * target_q

        # compute current Q
        current_q1 = self.critic1(obs, acts).squeeze(1)
        current_q2 = self.critic2(obs, acts).squeeze(1)
        # compute critic loss
        critic_loss1 = F.mse_loss(current_q1, target_q)
        critic_loss2 = F.mse_loss(current_q2, target_q)

        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        """
        Perturbation Loss
        """
        generated_action_ = self.cvae.decode(obs, z_device=self.device)
        perturbed_action_ = self.perturb(obs, generated_action_)
        perturbation_loss = -self.critic1(obs, perturbed_action_).mean()

        self.perturb_optim.zero_grad()
        perturbation_loss.backward()
        self.perturb_optim.step()

        """
        Update target networks
        """
        soft_target_update(self.critic1, self.target_critic1, tau=self.tau)
        soft_target_update(self.critic2, self.target_critic2, tau=self.tau)
        soft_target_update(self.perturb, self.target_perturb, tau=self.tau)

        self.train_step += 1

        train_summaries = {"cvae_loss": cvae_loss.cpu().item(),
                           "critic_loss_mean": ((critic_loss1 + critic_loss2) / 2).cpu().item(),
                           "perturbation_loss": perturbation_loss.cpu().item()}

        return train_summaries