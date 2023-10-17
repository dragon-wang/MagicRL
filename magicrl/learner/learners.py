from abc import abstractmethod, ABC
import torch
from magicrl.utils.logger import LearnLogger, AgentLogger
from magicrl.data.buffers import BaseBuffer
from magicrl.agents.base import BaseAgent
from magicrl.learner.exploration import explore_randomly, explore_by_agent
from magicrl.learner.evaluation import evaluate_agent, infer_agent
import numpy as np

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
                 **kwargs):
        super().__init__(**kwargs)

        self.explore_step = explore_step
    
    def learn(self):
        try:
            learner_logger = LearnLogger(self.learn_id, self.resume)
            agent_logger = AgentLogger(self.learn_id ,self.resume)
            if self.resume:
                agent_logger.load_agent(self.agent, -1)
                explore_by_agent(self.train_env, self.agent, self.buffer, self.explore_step)
            else:
                explore_randomly(self.train_env, self.buffer, self.explore_step)
            print("==============================start train===================================")
            obs, _ = self.train_env.reset()
            episode_reward = 0
            episode_length = 0

            # The main loop of "choose action -> act action -> add buffer -> train policy -> log data"
            while self.agent.train_step < self.max_train_step:
                act = self.agent.select_action(np.array(obs), eval=False)
                next_obs, rew, terminated, truncated, info = self.train_env.step(act)
                done = np.logical_or(terminated, truncated)
                episode_reward += rew
                transition = {"obs": obs,
                            "act": act,
                            "rew": rew,
                            "next_obs": next_obs,
                            "done": done}
                self.buffer.add(transition)
                obs = next_obs
                episode_length += 1

                batch = self.buffer.sample(device=self.agent.device)
                train_summaries = self.agent.train(batch)
                self.agent.train_step += 1

                if done:
                    self.agent.train_episode += 1
                    obs, _ = self.train_env.reset()
                
                    print(f"Time Step: {self.agent.train_step} Episode Num: {self.agent.train_episode}"
                        f"Episode Length: {episode_length} Episode Reward: {episode_reward:.2f}")
                    learner_logger.log_train_data({"episode_length": episode_length,
                                                        "episode_reward": episode_reward}, self.agent.train_step)
                    episode_reward = 0
                    episode_length = 0

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
            