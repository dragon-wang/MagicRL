from magicrl.agents.base import BaseAgent
from magicrl.agents.modelfree.dqn import DQNAgent
from magicrl.agents.modelfree.ddpg import DDPGAgent
from magicrl.agents.modelfree.sac import SACAgent
from magicrl.agents.modelfree.td3 import TD3Agent


__all__ = ['BaseAgent',
           'DQNAgent',
           'DDPGAgent',
           'SACAgent',
           'TD3Agent']
