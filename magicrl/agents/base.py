from abc import abstractmethod, ABC, ABCMeta
import torch
import numpy as np
import os


class BaseAgent(ABC):
    def __init__(self, gamma=0.99, device='cpu'):
        self.gamma = gamma
        self.device = torch.device(device)
        
        self.train_step = 0
        self.train_episode = 0

        # The attributions that needed to be saved by logger.AgentLogger
        self.attr_names = ["train_step", "train_episode"]

    @abstractmethod
    def select_action(self, obs, eval=False):
        """Select an action according to the observation

        Args:
            obs (_type_): The observation
            eval (bool): Whether used in evaluation
        """
        pass

    @abstractmethod
    def train(self, batch):
        """The main body of rl algorithm
        """
        pass

