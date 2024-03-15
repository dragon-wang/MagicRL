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
                 q_net: torch.nn.Module,
                 q_lr=0.001,
                 initial_eps=0.1,
                 end_eps=0.001,
                 eps_decay_period=2000,
                 eval_eps=0.001,
                 target_update_freq =10,
                 double_q=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.target_update_freq = target_update_freq
        self.double_q = double_q

        self.q_net = q_net.to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)

        # Decay for epsilon
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        self.eval_eps = eval_eps

        self.act_num = q_net.act_num

        self.attr_names.extend(['q_net', 'target_q_net', 'optim'])

    def select_action(self, obs, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.train_step + self.initial_eps, self.end_eps)

        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                return self.q_net(obs).argmax(dim=1).cpu().numpy()
        else:
            act = np.random.randint(0, self.act_num, size=obs.shape[0])
            return act

    def train(self, batch: dict):

        obs, acts, rews, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Compute target Q value
        with torch.no_grad():
            if self.double_q:
                # Double DQN
                next_acts = self.q_net(next_obs).max(dim=1)[1].unsqueeze(1)  # use Q net to get next actions, rather than target Q net
                target_q = rews + (1. - done) * self.gamma * self.target_q_net(next_obs).gather(1, next_acts).squeeze(1)
            else:
                # Vanilla DQN
                target_q = rews + (1. - done) * self.gamma * self.target_q_net(next_obs).max(dim=1)[0]

        # Compute current Q value
        current_q = self.q_net(obs).gather(1, acts.unsqueeze(-1).long()).squeeze(1)

        # Compute Q loss
        q_loss = 0.5 * (target_q - current_q).pow(2).mean()
        # Q_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the Q network
        self.optim.zero_grad()
        q_loss.backward()
        self.optim.step()

        # update target Q
        if self.train_step % self.target_update_freq == 0:
            hard_target_update(self.q_net, self.target_q_net)

        train_summaries = {"q_loss": q_loss.cpu().item(),
                           "q_mean": current_q.mean().cpu().item()}

        return train_summaries