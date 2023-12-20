from typing import Sequence, Type, Optional, List, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from magicrl.nn.common import MLP


"""
A complete feature net in MagicRL is like:
                                           h(rnns) 
visual -> |visual_net| ↘                  ↓                     ↗ Actor
                          feat_in -> |memory_net| -> h'(feat_out)
vector -> |vector_net| ↗                  ↓                     ↘ Critic
                                           h'(next_rnns)
In RNN and GRU, the rnns (rnn state) is a dict like: {'hidden': torch.tensor}
In LSTM, the rnns is a dict like: {'hidden': torch.tensor, 'cell': torch.tensor}
"""


class BaseFeatureNet(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        
        self.feature_dim = None


class AtariCNN(BaseFeatureNet):
    def __init__(self, num_frames_stack, pixel_size=[84, 84]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_frames_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.feature_dim = int(np.prod(self.net(torch.zeros(1, num_frames_stack, pixel_size[0], pixel_size[1])).shape[1:]))

    def forward(self, obs):
        return self.net(obs)


class SimpleRNN(BaseFeatureNet):
    def __init__(self, input_size, hidden_size, num_layers=1) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.feature_dim = hidden_size
        
    def forward(self, feat_in: torch.Tensor, rnns: dict[str, torch.Tensor]=None):
        """
        input size: feat_in: [N, L, D_in]          or [N, D_in]
                    rnns :   [N, num_layers, D_out]

        output size: feat_out: [N, L, D_out]          or [N, D_out]
                     rnns_out: [N, num_layers, D_out]
        
        Where 'N' is batch size, 'L' is the length of squence, and 'D_in' is the dimension of input feature.
        """
        if rnns is not None and not {"hidden"}.issubset(rnns.keys()):
            raise ValueError(
                f"Expected to find keys 'hidden' but instead found {rnns.keys()}",
            )
        
        if len(feat_in.shape) == 2:
            feat_in = feat_in.unsqueeze(1)  # [N, D_in] -> [N, 1, D_in]

        if rnns is None:
            feat_out, hidden = self.rnn(feat_in)
        else:
            # The rnns is stored with size: [N, num_layers, D_out], but rnn in pytorch need [num_layers, N, D_out].
            feat_out, hidden = self.rnn(feat_in, rnns['hidden'].transpose(0, 1).contiguous())
        
        return feat_out, {'hidden': hidden.transpose(0, 1)}


class SimpleLSTM(BaseFeatureNet):
    def __init__(self, input_size, hidden_size, num_layers=1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True)  # 可以是false，方便取l的最后一个作为隐状态
        
        self.feature_dim = hidden_size
        
    def forward(self, feat_in: torch.Tensor, rnns: dict[str, torch.Tensor]=None):
        if rnns is not None and not {"hidden", "cell"}.issubset(rnns.keys()):
            raise ValueError(
                f"Expected to find keys 'hidden' and 'cell' but instead found {rnns.keys()}",
            )
        
        if len(feat_in.shape) == 2:
            feat_in = feat_in.unsqueeze(1)  # [N, D_in] -> [N, 1, D_in]

        if rnns is None:
            feat_out, (hidden, cell) = self.rnn(feat_in)
        else:
            feat_out, (hidden, cell) = self.rnn(feat_in, 
                                        (rnns['hidden'].transpose(0, 1).contiguous(),
                                         rnns['cell'].transpose(0, 1).contiguous())) 
        
        return feat_out, {'hidden': hidden.transpose(0, 1),
                          'cell': cell.transpose(0, 1)}


class SimpleGRU(BaseFeatureNet):
    def __init__(self, input_size, hidden_size, num_layers=1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.feature_dim = hidden_size
        
    def forward(self, feat_in: torch.Tensor, rnns: dict[str, torch.Tensor]=None):
        if rnns is not None and not {"hidden"}.issubset(rnns.keys()):
            raise ValueError(
                f"Expected to find keys 'hidden' but instead found {rnns.keys()}",
            )
        
        if len(feat_in.shape) == 2:
            feat_in = feat_in.unsqueeze(1)  # [N, D_in] -> [N, 1, D_in]

        if rnns is None:
            feat_out, hidden = self.rnn(feat_in)
        else:
            feat_out, hidden = self.rnn(feat_in, rnns['hidden'].transpose(0, 1).contiguous())
        
        return feat_out, {'hidden': hidden.transpose(0, 1)}


class DRQNFeature(BaseFeatureNet):
    def __init__(self, num_frames_stack, pixel_size=[84, 84], rnn_hidden_size=512, num_layers=1) -> None:
        super().__init__()
        self.cnn = AtariCNN(num_frames_stack=num_frames_stack, pixel_size=pixel_size)
        self.lstm = SimpleLSTM(input_size=self.cnn.feature_dim, hidden_size=rnn_hidden_size, num_layers=num_layers)

        self.feature_dim = rnn_hidden_size

    def forward(self, obs, rnns):
        cnn_feat = self.cnn(obs)
        feat_out, rnns_out = self.lstm(cnn_feat, rnns)

        return feat_out, rnns_out