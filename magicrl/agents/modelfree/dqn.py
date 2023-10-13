import copy
import numpy as np
import torch
from magicrl.agents.base import BaseAgent
from magicrl.utils.train_tools import hard_target_update 


class DQNAgent(BaseAgent):
    """
    Implementation of Deep Q-Network (DQN)
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    def __init__(self,
                 Q_net: torch.nn.Module,
                 qf_lr=0.001,
                 initial_eps=0.1,
                 end_eps=0.001,
                 eps_decay_period=2000,
                 eval_eps=0.001,
                 target_update_freq =10,
                 action_dim = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.target_update_freq = target_update_freq

        self.Q_net = Q_net.to(self.device)
        self.target_Q_net = copy.deepcopy(self.Q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=qf_lr)

        self.attr_names = ["Q_net", "target_Q_net", "optimizer", "train_step", "train_episode"]

        # Decay for epsilon
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        self.eval_eps = eval_eps
        self.action_dim = action_dim

    def select_action(self, obs, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.train_step + self.initial_eps, self.end_eps)

        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                return int(self.Q_net(obs).argmax(dim=1).cpu())
        else:
            act = np.random.randint(0, self.action_dim)
            return act

    def train(self, batch: dict):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Compute target Q value
        with torch.no_grad():
            target_q = rews + (1. - done) * self.gamma * self.target_Q_net(next_obs).max(dim=1)[0]

        # Compute current Q value
        current_q = self.Q_net(obs).gather(1, acts.unsqueeze(-1).long()).squeeze(1)

        # Compute Q loss
        q_loss = 0.5 * (target_q - current_q).pow(2).mean()
        # Q_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # update target Q
        if self.train_step % self.target_update_freq == 0:
            hard_target_update(self.Q_net, self.target_Q_net)

        train_summaries = {"q_loss": q_loss.cpu().item()}

        return train_summaries