import copy
import torch
import torch.nn.functional as F

from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import soft_target_update 


class BEARAgent(BaseAgent):
    """
    Implementation of Bootstrapping Error Accumulation Reduction (BEAR)
    https://arxiv.org/abs/1906.00949
    BEAR's MMD Loss's weight alpha_prime is tuned automatically by default.

    Actor Loss: alpha_prime * MMD Loss + -minQ(s,a)
    Critic Loss: Like BCQ
    Alpha_prime Loss: -(alpha_prime * (MMD Loss - threshold))
    """
    def __init__(self,
                 actor: torch.nn.Module,
                 critic1: torch.nn.Module,
                 critic2: torch.nn.Module,
                 cvae: torch.nn.Module,
                 actor_lr=1e-4,
                 critic_lr=3e-4,
                 cvae_lr=3e-4,
                 tau=0.05,
                 lmbda=0.75,  # used for double clipped double q-learning
                 mmd_sigma=20.0,  # the sigma used in mmd kernel
                 kernel_type='gaussian',  # the type of mmd kernel(gaussian or laplacian)
                 lagrange_thresh=0.05,  # the hyper-parameter used in automatic tuning alpha in cql loss
                 n_action_samples=100,  # the number of action samples to compute the best action when choose action
                 n_target_samples=10,  # the number of action samples to compute BCQ-like target value
                 n_mmd_action_samples=4,  # the number of action samples to compute MMD
                 warmup_step=40000,  # do support matching with a warm start before policy(actor) train
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        # the network and optimizers
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.cvae = cvae.to(self.device)
        self.policy_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.cvae_optimizer = torch.optim.Adam(self.cvae.parameters(), lr=cvae_lr)

        self.tau = tau

        self.lmbda = lmbda
        self.mmd_sigma = mmd_sigma
        self.kernel_type = kernel_type
        self.lagrange_thresh = lagrange_thresh
        self.n_action_samples = n_action_samples
        self.n_target_samples = n_target_samples
        self.n_mmd_action_samples = n_mmd_action_samples
        self.warmup_step = warmup_step

        # mmd loss's temperature
        self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=1e-3)

        self.attr_names.extend(['actor', 'critic1', 'critic2', 'cvae', 
                                'policy_optim', 'critic_optim1', 'critic_optim2', 'cvae_optimizer'])

    def select_action(self, obs, eval=True):
        with torch.no_grad():
            obs_len = obs.shape[0]
            obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.repeat_interleave(obs, repeats=self.n_action_samples, dim=0)

            action, _, _ = self.actor(obs)
            q1 = self.critic1(obs, action)

            ind = q1.reshape(obs_len, self.n_action_samples, 1).argmax(1).squeeze().cpu()
            ind = ind + torch.arange(obs_len) * self.n_action_samples
            action = action[ind]
        return action.cpu().numpy()
    
    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def train(self, batch):
        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        """
        Train the Behaviour cloning policy to be able to take more than 1 sample for MMD.
        Conditional VAE is used as Behaviour cloning policy in BEAR.
        """
        recon_action, mu, log_std = self.cvae(obs, acts)
        cvae_loss = self.cvae.loss_function(recon_action, acts, mu, log_std)

        self.cvae_optimizer.zero_grad()
        cvae_loss.backward()
        self.cvae_optimizer.step()

        """
        Critic Training
        """
        with torch.no_grad():
            # generate 10 actions for every next_obs(Same as BCQ)
            next_obs = torch.repeat_interleave(next_obs, repeats=self.n_target_samples, dim=0).to(self.device)
            # compute target Q value of generated action
            target_q1 = self.target_critic1(next_obs, self.actor(next_obs)[0])
            target_q2 = self.target_critic2(next_obs, self.actor(next_obs)[0])
            # soft clipped double q-learning
            target_q = self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)
            # take max over each action sampled from the generation and perturbation model
            target_q = target_q.reshape(obs.shape[0], self.n_target_samples, 1).max(1)[0].squeeze(1)
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

        # MMD Loss
        # sample actions from dataset and current policy(B x N x D)
        raw_sampled_actions = self.cvae.decode_multiple_without_squash(obs, decode_num=self.n_mmd_action_samples,
                                                                           z_device=self.device)
        raw_actor_actions = self.actor.sample_multiple_without_squash(obs, sample_num=self.n_mmd_action_samples)
        if self.kernel_type == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        """
        Alpha prime training(lagrangian parameter update for MMD loss weight)
        """
        alpha_prime_loss = -(self.log_alpha_prime.exp() * (mmd_loss - self.lagrange_thresh)).mean()
        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss.backward(retain_graph=True)
        self.alpha_prime_optimizer.step()

        self.log_alpha_prime.data.clamp_(min=-5.0, max=10.0)  # clip for stability

        """
        Actor Training
        Actor Loss = alpha_prime * MMD Loss + -minQ(s,a)
        """
        a, log_prob, _ = self.actor(obs)
        min_q = torch.min(self.critic1(obs, a), self.critic2(obs, a)).squeeze(1)
        # policy_loss = (self.alpha * log_prob - min_q).mean()  # SAC Type
        policy_loss = - (min_q.mean())

        # BEAR Actor Loss
        actor_loss = (self.log_alpha_prime.exp() * mmd_loss).mean()
        if self.train_step > self.warmup_step:
            actor_loss = policy_loss + actor_loss
        self.policy_optim.zero_grad()
        actor_loss.backward()  # the mmd_loss will backward again in alpha_prime_loss.
        self.policy_optim.step()

        soft_target_update(self.critic1, self.target_critic1, tau=self.tau)
        soft_target_update(self.critic2, self.target_critic2, tau=self.tau)

        train_summaries = {"cvae_loss": cvae_loss.cpu().item(),
                           "actor_loss": policy_loss.cpu().item(),
                           "critic_loss_mean": ((critic_loss1 + critic_loss2) / 2).cpu().item(),
                           "alpha_prime_loss": alpha_prime_loss.cpu().item()}

        return train_summaries