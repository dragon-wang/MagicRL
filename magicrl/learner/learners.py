from abc import abstractmethod, ABC

import torch
import numpy as np

from magicrl.utils.logger import LearnLogger, AgentLogger
from magicrl.data import BaseBuffer
from magicrl.agents import BaseAgent
from magicrl.learner.exploration import explore_randomly, explore_by_agent
from magicrl.learner.evaluation import evaluate_agent, infer_agent
from magicrl.learner.collector import Collector

class LearnerBase(ABC):
    def __init__(self, 
                 learn_id,  # The name and path to save model and log tensorboard
                 train_env,  # The the environmrnt for train.
                 eval_env,  # The the environmrnt for evaluation.
                 agent: BaseAgent,  # The policy to be train.
                 buffer: BaseBuffer,  # The buffer to store the train data.
                 max_train_step, # The max train step
                 learner_log_freq,  # How often (time steps) the train data is logged.
                 agent_log_freq,  # How often (time steps) the policy is saved.
                 eval_freq, # How often (time steps) the policy is evaluated. it will not evaluate the agent during train if eval_freq < 0.
                 resume=False,
                 ) -> None:
        self.learn_id = learn_id
        self.train_env = train_env
        self.eval_env = eval_env
        self.agent = agent
        self.buffer = buffer
        self.max_train_step = max_train_step
        self.learner_log_freq = learner_log_freq
        self.agent_log_freq = agent_log_freq
        self.eval_freq = eval_freq
        self.resume = resume

        self.train_collector = Collector(self.train_env, self.agent, self.buffer)

    @abstractmethod
    def learn(self):
        pass  

    def inference(self, infer_env, episode_num):
        agent_logger = AgentLogger(self.learn_id ,True)
        agent_logger.load_agent(self.agent, -1)
        infer_agent(infer_env, self.agent, episode_num=episode_num)


class OffPolicyLearner(LearnerBase):
    def __init__(self,
                 explore_step,
                 batch_size,
                 **kwargs):
        super().__init__(**kwargs)

        self.explore_step = explore_step
        self.batch_size = batch_size

    def learn(self):
        try:
            learner_logger = LearnLogger(self.learn_id, self.resume)
            agent_logger = AgentLogger(self.learn_id ,self.resume)
            if self.resume:
                agent_logger.load_agent(self.agent, -1)
                self.train_collector.collect(n_step=self.explore_step, is_explore=True, random=False)
            else:
                self.train_collector.collect(n_step=self.explore_step, is_explore=True, random=True)

            print("==============================start train===================================")
            # The main loop of "choose action -> act action -> add buffer -> train policy -> log data"
            while self.agent.train_step < self.max_train_step:

                # collect data by interacting with the environment.
                self.train_collector.collect(n_step=1)

                # sample data from the buffer.
                batch = self.buffer.sample(self.batch_size, device=self.agent.device)

                # train agent with the sampled data.
                train_summaries = self.agent.train(batch)
                self.agent.train_step += 1

                # log train and evaluation data.
                if self.agent.train_step % self.learner_log_freq == 0:
                    learner_logger.log_train_data(train_summaries, self.agent.train_step)

                if self.eval_freq > 0 and self.agent.train_step % self.eval_freq == 0:
                    evaluate_summaries = evaluate_agent(self.eval_env, self.agent, episode_num=10)
                    print(evaluate_summaries)
                    learner_logger.log_eval_data(evaluate_summaries, self.agent.train_step)
                
                if self.agent.train_step % self.agent_log_freq == 0:
                    agent_logger.log_agent(self.agent, self.agent.train_step)

        except KeyboardInterrupt:
            print("Saving agent.......")
            agent_logger.log_agent(self.agent, self.agent.train_step)
            