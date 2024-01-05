import copy
import numpy as np
import torch
import torch.nn.functional as F

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class CQLAgent(BaseAgent):
    """
    Implementation of Conservative Q-Learning for Offline Reinforcement Learning (CQL)
    https://arxiv.org/abs/2006.04779
    This is CQL based on SAC, which is suitable for continuous action space.
    """
    def __init__(self,
                 actor: torch.nn.Module, 
                 critic1: torch.nn.Module, 
                 critic2: torch.nn.Module,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=0.05,
                 alpha=0.5,
                 auto_alpha_tuning=False,
                 min_q_weight=5.0,  # the value of alpha in CQL loss, set to 5.0 or 10.0 if not using lagrange
                 entropy_backup=False,  # whether use sac style target Q with entropy
                 max_q_backup=False,  # whether use max q backup
                 with_lagrange=False,  # whether auto tune alpha in Conservative Q Loss(different from the alpha in sac)
                 lagrange_thresh=0.0,  # the hyper-parameter used in automatic tuning alpha in cql loss
                 n_action_samples=10,  # the number of action sampled in importance sampling
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        self.act_dim = actor.act_dim

        # the network and optimizers
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.tau = tau
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning

        self.attr_names.extend(['actor','critic1', 'target_critic1', 'critic2', 'target_critic2', 
                                'actor_optim', 'critic_optim1', 'critic_optim2'])

        if self.auto_alpha_tuning:
            self.target_entropy = -self.act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=actor_lr)
            self.alpha = torch.exp(self.log_alpha)
            self.attr_names.extend(['log_alpha', 'alpha_optim'])

        # CQL
        self.min_q_weight = min_q_weight
        self.entropy_backup = entropy_backup
        self.max_q_backup = max_q_backup
        self.with_lagrange = with_lagrange
        self.lagrange_thresh = lagrange_thresh
        self.n_action_samples = n_action_samples

        if self.with_lagrange:
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optim = torch.optim.Adam([self.log_alpha_prime], lr=critic_lr)
            self.attr_names.extend(['log_alpha_prime', 'alpha_prime_optim'])

    def select_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            _, _, mu_action = self.actor(obs)
        return mu_action.cpu().numpy()

    def get_policy_actions(self, obs, n_action_samples):
        """
        get n*m actions from m obs
        :param obs: m obs
        :param n_action_samples: num of n
        """
        obs_temp = torch.repeat_interleave(obs, n_action_samples, dim=0).to(self.device)
        with torch.no_grad():
            actions, log_probs, _ = self.actor(obs_temp)
        return actions, log_probs.reshape(obs.shape[0], n_action_samples, 1)

    def get_actions_values(self, obs, actions, n_action_samples, q_net):
        """
        get n*m Q(s,a) from m obs and n*m actions
        :param obs: m obs
        :param actions: n actions
        :param n_action_samples: num of n
        :param q_net:
        """
        obs_temp = torch.repeat_interleave(obs, n_action_samples, dim=0).to(self.device)
        q = q_net(obs_temp, actions)
        q = q.reshape(obs.shape[0], n_action_samples, 1)
        return q

    def train(self, batch):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        """
        SAC Loss
        """
        # compute policy Loss
        a, log_prob, _ = self.actor(obs)
        min_q = torch.min(self.critic1(obs, a), self.critic2(obs, a)).squeeze(1)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        # compute Q Loss
        q1 = self.critic1(obs, acts).squeeze(1)
        q2 = self.critic2(obs, acts).squeeze(1)
        with torch.no_grad():
            if not self.max_q_backup:
                next_a, next_log_prob, _ = self.actor(next_obs)
                min_target_next_q = torch.min(self.target_critic1(next_obs, next_a),
                                              self.target_critic2(next_obs, next_a)).squeeze(1)
                if self.entropy_backup:
                    # y = rews + self.gamma * (1. - done) * (min_target_next_q - self.alpha * next_log_prob)
                    min_target_next_q = min_target_next_q - self.alpha * next_log_prob
            else:
                """when using max q backup"""
                next_a_temp, _ = self.get_policy_actions(next_obs, n_action_samples=10)
                target_qf1_values = self.get_actions_values(next_obs, next_a_temp, self.n_action_samples, self.critic1).max(1)[0]
                target_qf2_values = self.get_actions_values(next_obs, next_a_temp, self.n_action_samples, self.critic2).max(1)[0]
                min_target_next_q = torch.min(target_qf1_values, target_qf2_values).squeeze(1)

            y = rews + self.gamma * (1. - done) * min_target_next_q

        critic_loss1 = F.mse_loss(q1, y)
        critic_loss2 = F.mse_loss(q2, y)

        """
        CQL Loss
        Total Loss = SAC loss + min_q_weight * CQL loss
        """
        # Use importance sampling to compute log sum exp of Q(s, a), which is shown in paper's Appendix F.
        random_sampled_actions = torch.FloatTensor(obs.shape[0] * self.n_action_samples, acts.shape[-1]).uniform_(-1, 1).to(self.device)
        curr_sampled_actions, curr_log_probs = self.get_policy_actions(obs, self.n_action_samples)
        # This is different from the paper because it samples not only from the current state, but also from the next state
        next_sampled_actions, next_log_probs = self.get_policy_actions(next_obs, self.n_action_samples)
        q1_rand = self.get_actions_values(obs, random_sampled_actions, self.n_action_samples, self.critic1)
        q2_rand = self.get_actions_values(obs, random_sampled_actions, self.n_action_samples, self.critic2)
        q1_curr = self.get_actions_values(obs, curr_sampled_actions, self.n_action_samples, self.critic1)
        q2_curr = self.get_actions_values(obs, curr_sampled_actions, self.n_action_samples, self.critic2)
        q1_next = self.get_actions_values(obs, next_sampled_actions, self.n_action_samples, self.critic1)
        q2_next = self.get_actions_values(obs, next_sampled_actions, self.n_action_samples, self.critic2)

        random_density = np.log(0.5 ** acts.shape[-1])

        cat_q1 = torch.cat([q1_rand - random_density, q1_next - next_log_probs, q1_curr - curr_log_probs], dim=1)
        cat_q2 = torch.cat([q2_rand - random_density, q2_next - next_log_probs, q2_curr - curr_log_probs], dim=1)

        min_qf1_loss = torch.logsumexp(cat_q1, dim=1).mean()
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1).mean()

        min_qf1_loss = self.min_q_weight * (min_qf1_loss - q1.mean())
        min_qf2_loss = self.min_q_weight * (min_qf2_loss - q2.mean())

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1e6)
            # the lagrange_thresh has no effect on the gradient of policy,
            # but it has an effect on the gradient of alpha_prime
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.lagrange_thresh)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.lagrange_thresh)

            alpha_prime_loss = -(min_qf1_loss + min_qf2_loss) * 0.5

            self.alpha_prime_optim.zero_grad()
            alpha_prime_loss.backward(retain_graph=True)  # the min_qf_loss will backward again latter, so retain graph.
            self.alpha_prime_optim.step()
        else:
            alpha_prime_loss = torch.tensor(0)

        critic_loss1 = critic_loss1 + min_qf1_loss
        critic_loss2 = critic_loss2 + min_qf2_loss

        """
        Update networks
        """
        # Update policy network parameter
        # policy network's update should be done before updating q network, or there will make some errors
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update q network1 parameter
        self.critic_optim1.zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic_optim1.step()

        # Update q network2 parameter
        self.critic_optim2.zero_grad()
        critic_loss2.backward(retain_graph=True)
        self.critic_optim2.step()

        if self.auto_alpha_tuning:
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
                           "alpha_loss": alpha_loss.cpu().item(),
                           "alpha_prime_loss": alpha_prime_loss.cpu().item()}

        return train_summaries
