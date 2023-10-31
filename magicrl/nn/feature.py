from typing import Sequence, Type, Optional, List, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from magicrl.nn.common import MLP


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