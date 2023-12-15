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
