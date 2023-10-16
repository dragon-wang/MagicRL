import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Type, Optional, List, Union
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from magicrl.nn.common import MLP


class MLPQsaNet(nn.Module):
    """
    DDPG Critic, SAC Q net, BCQ Critic
    Input (s,a), output Q(s,a)
    """
    def __init__(self, obs_dim, act_dim, hidden_size, hidden_activation=nn.ReLU):
        super(MLPQsaNet, self).__init__()
        self.mlp = MLP(input_dim=obs_dim + act_dim,
                       output_dim=1,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        q = self.mlp(x)
        return q


class DDPGMLPActor(nn.Module):
    """
    DDPG Actor
    """
    def __init__(self, obs_dim, act_dim, act_bound, hidden_size, hidden_activation=nn.ReLU):
        super(DDPGMLPActor, self).__init__()
        self.mlp = MLP(input_dim=obs_dim, output_dim=act_dim,
                       hidden_size=hidden_size, hidden_activation=hidden_activation)
        self.act_bound = act_bound
        self.act_dim = act_dim

    def forward(self, obs):
        a = torch.tanh(self.mlp(obs))
        a = self.act_bound * a
        return a
    

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLPSquashedReparamGaussianPolicy(nn.Module):
    """
    Policy net. Used in SAC, CQL, BEAR.
    Input s, output reparameterized, squashed action and log probability of this action
    """
    def __init__(self, obs_dim, act_dim, act_bound, hidden_size, hidden_activation=nn.ReLU, edge=3e-3):

        super(MLPSquashedReparamGaussianPolicy, self).__init__()

        self.mlp = MLP(input_dim=obs_dim, output_dim=hidden_size[-1], hidden_size=hidden_size[:-1],
                       hidden_activation=hidden_activation, output_activation=hidden_activation)
        self.fc_mu = nn.Linear(hidden_size[-1], act_dim)
        self.fc_log_std = nn.Linear(hidden_size[-1], act_dim)

        # self.fc_mu.weight.data.uniform_(-edge, edge)
        # self.fc_log_std.bias.data.uniform_(-edge, edge)

        self.hidden_activation = hidden_activation

        self.act_dim = act_dim
        self.act_bound = act_bound

    def forward(self, obs):
        x = self.mlp(obs)

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        action = self.act_bound * a
        log_prob = torch.sum(dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6), dim=1)
        mu_action = self.act_bound * torch.tanh(mu)  # used in evaluation

        return action, log_prob, mu_action

    def sample_multiple_without_squash(self, obs, sample_num):
        x = self.mlp(obs)
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        raw_action = dist.rsample((sample_num, ))

        return raw_action.transpose(0, 1)  # N x B X D -> B x N x D (N:sample num, B:batch size, D:action dim)