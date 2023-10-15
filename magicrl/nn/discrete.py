import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from magicrl.nn.common import MLP


class MLPQsNet(nn.Module):
    """
    DQN Q net
    Input s, output all Q(s, a)
    """
    def __init__(self, obs_dim, act_num, hidden_size, hidden_activation=nn.ReLU):
        super(MLPQsNet, self).__init__()
        self.mlp = MLP(input_dim=obs_dim,
                       output_dim=act_num,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)
        
        self.act_num = act_num

    def forward(self, obs):
        return self.mlp(obs)