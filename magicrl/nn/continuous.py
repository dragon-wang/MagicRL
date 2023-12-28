from typing import Sequence, Type, Optional, List, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from magicrl.nn.common import MLP
from magicrl.nn.feature import BaseFeatureNet


LOG_STD_MIN = -20
LOG_STD_MAX = 2


"""
The network in MagicRL is like:
    obs -> feature_net -> mlp -> act or Q
"""
class SimpleActor(nn.Module):
    """SimpleActor used in DDPG, TD3.
    """
    def __init__(self, 
                 obs_dim: Union[int, list, np.ndarray], 
                 act_dim: int, 
                 act_bound: float, 
                 hidden_size: List[int], 
                 hidden_activation=nn.ReLU,
                 feature_net: BaseFeatureNet=None) -> None:
        super().__init__()
        
        feature_dim = obs_dim if feature_net is None else feature_net.feature_dim

        self.mlp = MLP(input_dim=feature_dim, output_dim=act_dim, hidden_size=hidden_size,
                       hidden_activation=hidden_activation)
        
        self.feature_net = feature_net

        self.act_bound = act_bound
        self.act_dim = act_dim

    def forward(self, obs: Union[np.ndarray, torch.Tensor]):
        act = self.mlp(obs) if self.feature_net is None else self.mlp(self.feature_net(obs))
        act = self.act_bound * torch.tanh(act)
        
        return act
    
class SimpleCritic(nn.Module):
    """SimpleCritic used in DDPG, TD3, SAC, PPO.
    """
    def __init__(self, 
                 obs_dim: Union[int, list, np.ndarray], 
                 act_dim: int = 0, 
                 hidden_size: List[int] = [], 
                 hidden_activation = nn.ReLU,
                 feature_net: BaseFeatureNet = None) -> None:
        super().__init__()

        feature_dim = obs_dim if feature_net is None else feature_net.feature_dim

        self.mlp = MLP(input_dim=feature_dim + act_dim, 
                       output_dim=1, 
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)
        
        self.feature_net = feature_net

    def forward(self, 
                obs: Union[np.ndarray, torch.Tensor], 
                act: Union[np.ndarray, torch.Tensor, None] = None):
        
        feature = obs if self.feature_net is None else self.feature_net(obs)

        if act is not None:
            feature = torch.cat([feature, act], dim=1)
            
        q = self.mlp(feature)

        return q
        

class RepapamGaussionActor(nn.Module):
    def __init__(self, 
                 obs_dim: Union[int, list, np.ndarray], 
                 act_dim: int, 
                 act_bound: float, 
                 hidden_size: List[int], 
                 hidden_activation=nn.ReLU,
                 feature_net: BaseFeatureNet=None) -> None:
        super().__init__()
        
        feature_dim = obs_dim if feature_net is None else feature_net.feature_dim

        self.mlp = MLP(input_dim=feature_dim, output_dim=hidden_size[-1], hidden_size=hidden_size[:-1],
                       hidden_activation=hidden_activation, output_activation=hidden_activation)
        
        self.fc_mu = nn.Linear(hidden_size[-1], act_dim)
        self.fc_log_std = nn.Linear(hidden_size[-1], act_dim)
        
        self.feature_net = feature_net

        self.act_bound = act_bound
        self.act_dim = act_dim

        self._eps = np.finfo(np.float32).eps.item()

    def forward(self, obs):
        x =  self.mlp(obs) if self.feature_net is None else self.mlp(self.feature_net(obs))

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        raw_act = dist.rsample()
        squashed_act = torch.tanh(raw_act)

        act = self.act_bound * squashed_act
        log_prob = torch.sum(dist.log_prob(raw_act) - torch.log(1 - squashed_act.pow(2) + self._eps), dim=1)

        mu_act = self.act_bound * torch.tanh(mu)  # used in evaluation and inference.

        return act, log_prob, mu_act

    def sample_multiple_without_squash(self, obs, sample_num):
        x =  self.mlp(obs) if self.feature_net is  None else self.mlp(self.feature_net(obs))

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        raw_act = dist.rsample((sample_num, ))

        return raw_act.transpose(0, 1)  # N x B X D -> B x N x D (N:sample num, B:batch size, D:action dim)


class GaussionActor(nn.Module):
    def __init__(self, 
                 obs_dim: Union[int, list, np.ndarray], 
                 act_dim: int, 
                 act_bound: float, 
                 hidden_size: List[int], 
                 hidden_activation=nn.Tanh,
                 feature_net: BaseFeatureNet=None) -> None:
        super().__init__()
        
        feature_dim = obs_dim if feature_net is None else feature_net.feature_dim

        self.mlp = MLP(input_dim=feature_dim, output_dim=hidden_size[-1], hidden_size=hidden_size[:-1],
                       hidden_activation=hidden_activation, output_activation=hidden_activation)
        
        self.fc_mu = nn.Linear(hidden_size[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim), requires_grad=True)
        
        self.feature_net = feature_net

        self.act_bound = act_bound
        self.act_dim = act_dim


    def forward(self, obs):
        x =  self.mlp(obs) if self.feature_net is None else self.mlp(self.feature_net(obs))

        mu = self.fc_mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        act = dist.sample().clip(-self.act_bound, self.act_bound)
        mu_act = mu.clip(-self.act_bound, self.act_bound)

        return act, mu_act
    
    def get_logprob_entropy(self, obs, act):
        x =  self.mlp(obs) if self.feature_net is None else self.mlp(self.feature_net(obs))

        mu = self.fc_mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        log_prob = dist.log_prob(act).sum(1)
        entropy = dist.entropy().sum(1)

        return log_prob, entropy  # act: (n, m) -> log_prob: (n, ); entropy: (n, )
    

class CVAE(nn.Module):
    """
    Conditional Variational Auto-Encoder(CVAE) used in BCQ and PLAS
    ref: https://github.com/sfujim/BCQ/blob/4876f7e5afa9eb2981feec5daf67202514477518/continuous_BCQ/BCQ.py#L57
    """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 latent_dim,
                 act_bound):
        """
        :param obs_dim: The dimension of observation
        :param act_dim: The dimension if action
        :param latent_dim: The dimension of latent in CVAE
        :param act_bound: The maximum value of the action
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.act_bound = act_bound

        # encoder net
        self.e1 = nn.Linear(obs_dim + act_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.e3_mu = nn.Linear(750, latent_dim)
        self.e3_log_std = nn.Linear(750, latent_dim)

        # decoder net
        self.d1 = nn.Linear(obs_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, act_dim)

    def encode(self, obs, action):
        h1 = F.relu(self.e1(torch.cat([obs, action], dim=1)))
        h2 = F.relu(self.e2(h1))
        mu = self.e3_mu(h2)
        log_std = self.e3_log_std(h2).clamp(-4, 15)  # Clamped for numerical stability
        return mu, log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # sample from standard normal distribution
        z = mu + eps * std
        return z

    def decode(self, obs, z=None, z_device=torch.device('cpu')):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(z_device).clamp(-0.5, 0.5)
        h4 = F.relu(self.d1(torch.cat([obs, z], dim=1)))
        h5 = F.relu(self.d2(h4))
        recon_action = torch.tanh(self.d3(h5)) * self.act_bound
        return recon_action

    def decode_multiple_without_squash(self, obs, decode_num=10, z=None, z_device=torch.device('cpu')):
        """
        decode n*b action from b obs and not squash
        """
        if z is None:
            z = torch.randn((obs.shape[0] * decode_num, self.latent_dim)).to(z_device).clamp(-0.5, 0.5)
        obs_temp = torch.repeat_interleave(obs, decode_num, dim=0)
        h4 = F.relu(self.d1(torch.cat([obs_temp, z], dim=1)))
        h5 = F.relu(self.d2(h4))
        raw_action = self.d3(h5)
        return raw_action.reshape(obs.shape[0], decode_num, -1)  # B*N x D -> B x N x D

    def forward(self, obs, action):
        mu, log_std = self.encode(obs, action)
        z = self.reparametrize(mu, log_std)
        # std = torch.exp(log_std)
        # dist = Normal(mu, std)
        # z = dist.rsample()
        recon_action = self.decode(obs, z)
        return recon_action, mu, log_std

    def loss_function(self, recon, action, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std)).mean()
        loss = recon_loss + 0.5 * kl_loss
        return loss


class PerturbActor(nn.Module):
    """Used in BCQ.
    """
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 act_bound, 
                 hidden_size, 
                 hidden_activation=nn.ReLU,
                 phi=0.05  # the Phi in perturbation model:
                 ):
        super().__init__()

        self.mlp = MLP(input_dim=obs_dim + act_dim,
                       output_dim=act_dim,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)

        self.act_bound = act_bound
        self.phi = phi

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        a = torch.tanh(self.mlp(x))
        a = self.phi * self.act_bound * a
        return (a + action).clamp(-self.act_bound, self.act_bound)