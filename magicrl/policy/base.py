from abc import abstractmethod, ABC, ABCMeta
import torch
import numpy as np
import os


class PolicyBase(ABC):
    def __init__(self,
                 gamma, # The decay factor
                 eval_freq, # How often (time steps) the policy is evaluated during train. it will not evaluate the agent if eval_freq < 0.
                 max_train_step, # The max train step
                 train_id, # The name and path to save model and log tensorboard
                 log_interval, # The number of steps taken to record the model and the tensorboard
                 resume, # Whether load the last saved model and continue to train
                 device,  # The device. Choose cpu or cuda
                 ):
        self.gamma = gamma
        self.eval_freq = eval_freq
        self.max_train_step = max_train_step
        self.train_id = train_id
        self.log_interval = log_interval
        self.resume = resume
        self.device = torch.device(device)

        self.train_step = 0

        # self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", self.train_id)
        # self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
    
    @abstractmethod
    def choose_action(self, obs, eval=False):
        """Select an action according to the observation

        Args:
            obs (_type_): The observation
            eval (bool): Whether used in evaluation
        """
        pass

    @abstractmethod
    def train(self):
        """The main body of rl algorithm
        """
        pass

