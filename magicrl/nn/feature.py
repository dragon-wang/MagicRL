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
    def __init__(self, feature_dim) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim