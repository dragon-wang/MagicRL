from abc import abstractmethod, ABC

import torch
import numpy as np

from magicrl.utils.logger import LearnLogger, AgentLogger
from magicrl.data import BaseBuffer
from magicrl.agents import BaseAgent
from magicrl.learner.interactor import Collector, Evaluator

class BaseLearner(ABC):
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

        self.collector = Collector(self.train_env, self.agent, self.buffer)
        self.evaluator = Evaluator(self.eval_env, self.agent)

    @abstractmethod
    def learn(self):
        pass  


class OffPolicyLearner(BaseLearner):
    def __init__(self,
                 explore_step,
                 batch_size,
                 collect_per_step=1,  # How many transitions the agent collect in every step.
                 **kwargs):
        super().__init__(**kwargs)

        self.explore_step = explore_step
        self.batch_size = batch_size
        self.collect_per_step = collect_per_step

    def learn(self):
        try:
            learner_logger = LearnLogger(self.learn_id, self.resume)
            agent_logger = AgentLogger(self.learn_id ,self.resume)
            if self.resume:
                agent_logger.load_agent(self.agent, -1)
                self.collector.collect(n_step=self.explore_step, is_explore=True, random=False)
            else:
                self.collector.collect(n_step=self.explore_step, is_explore=True, random=True)

            print("==============================start train===================================")
            # The main loop of "choose action -> act action -> add buffer -> train policy -> log data"
            while self.agent.train_step < self.max_train_step:

                # collect data by interacting with the environment.
                self.collector.collect(n_step=self.collect_per_step)

                # sample data from the buffer.
                batch = self.buffer.sample(self.batch_size, device=self.agent.device)

                # train agent with the sampled data.
                train_summaries = self.agent.train(batch)
                self.agent.train_step += 1  # the meaning of 'train_step' here is the same as 'time_step'.

                # log train data.
                if self.agent.train_step % self.learner_log_freq == 0:
                    learner_logger.log_train_data(train_summaries, self.agent.train_step)

                # log evaluate data.
                if self.eval_freq > 0 and self.agent.train_step % self.eval_freq == 0:
                    evaluate_summaries = self.evaluator.evaluate()
                    learner_logger.log_eval_data(evaluate_summaries, self.agent.train_step)
                
                # log trained agent.
                if self.agent.train_step % self.agent_log_freq == 0:
                    agent_logger.log_agent(self.agent, self.agent.train_step)

        except KeyboardInterrupt:
            print("Saving agent.......")
            agent_logger.log_agent(self.agent, self.agent.train_step)


class OnPolicyLearner(BaseLearner):
    def __init__(self,
                 trajectory_length=128,
                 **kwargs):
        super().__init__(**kwargs)

        self.trajectory_length = trajectory_length

    def learn(self):
        try:
            learner_logger = LearnLogger(self.learn_id, self.resume)
            agent_logger = AgentLogger(self.learn_id ,self.resume)
            if self.resume:
                agent_logger.load_agent(self.agent, -1)

            print("==============================start train===================================")
            # The main loop of "choose action -> act action -> add buffer -> train policy -> log data"
            t_length = 0
            while self.agent.train_step < self.max_train_step:

                # collect data by interacting with the environment.
                self.collector.collect(n_step=1)
                t_length += 1

                if t_length == self.trajectory_length:
                    # sample data from the buffer and clear the buffer.
                    self.buffer.finish_path(agent=self.agent)
                    batch = self.buffer.sample(self.trajectory_length, device=self.agent.device)       
                    # train agent with the sampled data.
                    train_summaries = self.agent.train(batch)

                    self.buffer.clear()
                    t_length = 0

                self.agent.train_step += 1  # the meaning of 'train_step' here is the same as 'time_step'.

                # log train data.
                if self.agent.train_step % self.learner_log_freq == 0:
                    learner_logger.log_train_data(train_summaries, self.agent.train_step)

                # log evaluate data.
                if self.eval_freq > 0 and self.agent.train_step % self.eval_freq == 0:
                    evaluate_summaries = self.evaluator.evaluate()
                    learner_logger.log_eval_data(evaluate_summaries, self.agent.train_step)
                
                # log trained agent.
                if self.agent.train_step % self.agent_log_freq == 0:
                    agent_logger.log_agent(self.agent, self.agent.train_step)

        except KeyboardInterrupt:
            print("Saving agent.......")
            agent_logger.log_agent(self.agent, self.agent.train_step)