import os
import torch
from torch.utils.tensorboard import SummaryWriter


# The directory of MagicRL. e.g. /xxx/xxx/MagicRL
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The directory of log results. e.g. /xxx/xxx/MagicRL/results
LOG_DIR = os.path.join(ROOT_DIR, "results")

# Recursively create a directory
def _make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


# Remove all files in a directory.
def _clean_dir(dir):
    if os.path.exists(dir):
        files = os.listdir(dir)
        for file in files:
            os.remove(os.path.join(dir, file))


class LearnLogger:
    def __init__(self, learn_id, resume):
        self.learn_log_dir = os.path.join(LOG_DIR, learn_id, "learn_log")
        if not resume:
            _clean_dir(self.learn_log_dir)
            _make_dir(self.learn_log_dir)
        self.writer = SummaryWriter(self.learn_log_dir)

    def log_train_data(self, log_datas: dict, step):
        for log_data in log_datas.items():
            self.writer.add_scalar("train_data/" + log_data[0], log_data[1], step)
        self.writer.flush()

    def log_eval_data(self, log_datas: dict, step):
        for log_data in log_datas.items():
            self.writer.add_scalar("evaluate_data/" + log_data[0], log_data[1], step)
        self.writer.flush()


class AgentLogger:
    def __init__(self, learn_id, resume):
        self.agent_log_dir = os.path.join(LOG_DIR, learn_id, "agent_log")
        if not resume:
            _clean_dir(self.agent_log_dir)
            _make_dir(self.agent_log_dir)

    def log_agent(self, agent, step):
        checkpoint_path = os.path.join(self.agent_log_dir, "checkpoint_" + str(step))
        checkpoint = {}
        for attr_name in agent.attr_names:
            checkpoint[attr_name] = getattr(agent, attr_name)
        torch.save(checkpoint, checkpoint_path)
        print("The agent is saved in ", checkpoint_path)

    def load_agent(self, agent, step):
        checkpoint_path = os.path.join(self.agent_log_dir, "checkpoint_" + str(step))
        checkpoint = torch.load(checkpoint_path)
        for attr_name in agent.attr_names:
            setattr(agent, attr_name, checkpoint[attr_name])
        print("The agent is loaded from ", checkpoint_path)

if __name__=="__main__":
    print(ROOT_DIR)
    _clean_dir("/Users/wt/Documents/PythonProjects/MagicRL/results")