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