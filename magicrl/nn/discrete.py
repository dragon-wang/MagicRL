from typing import Sequence, Type, Optional, List, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from magicrl.nn import MLP, BaseFeatureNet


class QNet(nn.Module):
    """
    DQN Q net
    Input s, output all Q(s, a)
    """
    def __init__(self, 
                 obs_dim: Union[int, list, np.ndarray], 
                 act_num: int, 
                 hidden_size, 
                 hidden_activation=nn.ReLU,
                 feature_net: BaseFeatureNet = None):
        super().__init__()

        self.act_num = act_num

        feature_dim = obs_dim if feature_net is None else feature_net.feature_dim

        self.mlp = MLP(input_dim=feature_dim,
                       output_dim=act_num,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)
        
        self.feature_net = feature_net
        

    def forward(self, obs):
        feature = self.feature_net(obs) if self.feature_net is not None else obs
        q = self.mlp(feature)
        return q