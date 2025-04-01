from abc import abstractmethod, ABC, ABCMeta
from typing import Sequence, Type, Optional, List, Union, Dict, Tuple

import torch
import numpy as np


def _build_buffer(buffer: Dict, transition: Dict, buffer_size):
    """Initial the buffer from a transition dict sampled from the environment.

    The transition may be like:
    "obs": "vector": "vector_0": np.ndarray (a1,)
                     "vector_1": np.ndarray (a2,)
                     ...
           "visual": "visual_0": np.ndarray (b1, c1)
                     "visual_1": np.ndarray (b2, c2)
                     ...
    "act": "discrete": np.ndarray ()  # scalar
           "continuous": np.ndarray (d1, )
    "rew": np.ndarray ()  # scalar
    "done": np.ndarray ()  # scalar
    "rnn_state": "hidden": np.ndarray (num_layers, rnn_hidden_size)
                 "cell": np.ndarray (num_layers, rnn_hidden_size)

    If buffer_size = n, the buffer is initialized like:
    "obs": "vector": "vector_0": np.empty (n, a1)
                     "vector_1": np.empty (n, a2)
                     ...
           "visual": "visual_0": np.empty (n, b1, c1)
                     "visual_1": np.empty (n, b2, c2)
                     ...
    "act": "discrete": np.empty (n, )
           "continuous": np.empty (n, d1)
    "rew": np.empty (n, )  # scalar
    "done": np.empty (n, )  # scalar
    "rnn_state": "hidden": np.ndarray (n, num_layers, rnn_hidden_size)
                 "cell": np.ndarray (n, num_layers, rnn_hidden_size)

    *Note: In order to reduce memory consumption, the "next_obs" is stored in "ReplayBuffer".
           But it is stored in "TrajectoryBuffer", because the "TrajectoryBuffer" will be cleared after each training step.

    Args:
        buffer (Dict): The buffer that need to build.
        transition (Dict): A transition sampled from the environment.
        buffer_size (_type_): The maximum number of transitions that a buffer can store.
    """
    for k, v in transition.items():
        if not isinstance(v, Dict):
            v = np.array(v)
            buffer[k] = np.empty((buffer_size, ) + v.shape, v.dtype)
        else:
            buffer[k] = {}
            _build_buffer(buffer[k], transition[k], buffer_size)


def _add_tran(buffer: Dict, transition: Dict, index=1):
    """Add a transition into buffer.
    
    If the buffer with buffer_size = 5 is:
    "obs": "vector": "vector_0": [a, 0, 0, 0, 0]
                     "vector_1": [b, 0, 0, 0, 0]
    "act":  [c, 0, 0, 0, 0]
    "rew":  [d, 0, 0, 0, 0]
    "done": [e, 0, 0, 0, 0]

    And the transition is:
    "obs": "vector": "vector_0": f
                     "vector_1": g
    "act":  h
    "rew":  i
    "done": j

    After call _add_tran(buffer, transition, 1), the buffer is:
    "obs": "vector": "vector_0": [a, f, 0, 0, 0]
                     "vector_1": [b, g, 0, 0, 0]
    "act":  [c, h, 0, 0, 0]
    "rew":  [d, i, 0, 0, 0]
    "done": [e, j, 0, 0, 0]

    Args:
        buffer (Dict): The buffer that need to add.
        transition (Dict): A transition sampled from the environment.
        index (int): the index to add in the buffer.
    """
    for k, v in transition.items():
        if not isinstance(v, Dict):
            buffer[k][index] = transition[k]
        else:
            _add_tran(buffer[k], transition[k], index)    


def _get_trans(buffer: Dict, index: Union[int, Sequence[int]]):
    """Get one or several transitions from the buffer according to the index.
    If index is i, the result will squeeze the dim;
    If index is [i] or [i,j], the result will not squeeze the dim;

    If the buffer with buffer_size = 3 is:
    "obs": [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
    "act": [[d1, d2, d3], [e1, e2, e3], [f1, f2, f3]]

    The result of _get_trans(buffer, 1) is:
    "obs": [b1, b2, b3]
    "act": [e1, e2, e3]

    The result of _get_trans(buffer, [1]) is:
    "obs": [[b1, b2, b3]]
    "act": [[e1, e2, e3]]

    The result of _get_trans(buffer, [1,2]) is:
    "obs": [[b1, b2, b3], [c1, c2, c3]]
    "act": [[e1, e2, e3], [f1, f2, f3]] 

    Args:
        buffer (Dict): The buffer that need to add.
        index (Union[int, Sequence[int]]): the index(es) to get in the buffer.

    Returns:
        _type_: The obtained transitions. 
    """
    trans = {}
    for k, v in buffer.items():
        if not isinstance(v, Dict):
            trans[k] = v[index]
        else:
            trans[k]  = _get_trans(buffer[k], index)
    return trans


def _to_tensor(data: Dict, device, dtype=torch.float32):
    """Convert all ndarrays in buffer or batch to tensors.

    Args:
        batch (Dict): the buffer or batch that need to convert.
        device (_type_): the device in torch.
        dtype (_type_): the dtype of tensor.
    """
    for k, v in data.items():
        if not isinstance(v, Dict):
            data[k] = torch.as_tensor(data[k], dtype=dtype, device=device)
        else:
            _to_tensor(data[k], device)

def _pad_buffer(buffer: Dict, pad_size: int):
    """Pad buffer with zeros.

    If the buffer with buffer_size = 5 is:
    "obs": "vector": "vector_0": [a, 0, 0, 0, 0]
                     "vector_1": [b, 0, 0, 0, 0]
    "act":  [c, 0, 0, 0, 0]
    "rew":  [d, 0, 0, 0, 0]
    "done": [e, F, F, F, F]

    After call _pad_buffer(buffer, 2), the buffer is:
    "obs": "vector": "vector_0": [a, 0, 0, 0, 0, 0, 0]
                     "vector_1": [b, 0, 0, 0, 0, 0, 0]
    "act":  [c, 0, 0, 0, 0, 0, 0]
    "rew":  [d, 0, 0, 0, 0, 0, 0]
    "done": [e, F, F, F, F, F, F]

    The new buffer_size is 7.
    """
    for k, v in buffer.items():
        if not isinstance(v, Dict):
            shp = (pad_size, *buffer[k].shape[1:])
            buffer[k] = np.concatenate((buffer[k], np.zeros(shp, dtype=buffer[k].dtype)), axis=0)
        else:
            _pad_buffer(buffer[k], pad_size)


def _concat_batch(batch_list: Union[List[Dict], Tuple[Dict]]):
    batches = {}
    for k, v in batch_list[0].items():
        if not isinstance(v, Dict):
            batches[k] = np.concatenate([batch_list[i][k] for i in range(len(batch_list))])
        else: 
            batches[k] = _concat_batch([batch_list[i][k] for i in range(len(batch_list))])
    return batches

def _stack_batch(batch_list: Union[List[Dict], Tuple[Dict]]):
    batches = {}
    for k, v in batch_list[0].items():
        if not isinstance(v, Dict):
            batches[k] = np.stack([batch_list[i][k] for i in range(len(batch_list))])
        else: 
            batches[k] = _stack_batch([batch_list[i][k] for i in range(len(batch_list))])
    return batches


class BaseBuffer(ABC):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self._buffer = {}
        self._pointer = 0  # Point to the current position in the buffer
        self._current_size = 0  # The current size of the buffer
        self.episode_steps = np.zeros(self.buffer_size, dtype=np.int32)
    
    @abstractmethod
    def add(self, transition: Dict, step: int):
        pass
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def save(self):
        # save to hdf5
        pass

    @abstractmethod
    def load(self):
        # load from hdf5
        pass

class ReplayBuffer(BaseBuffer):
    def __init__(self, buffer_size: int = 0):
        super().__init__(buffer_size)
        
    def add(self, transition: Dict, step: int):
        if not self._buffer:
            _build_buffer(self._buffer, transition, self.buffer_size)
        _add_tran(self._buffer, transition, index=self._pointer)
        self.episode_steps[self._pointer] = step
        
        self._pointer = (self._pointer + 1) % self.buffer_size
        self._current_size = min(self._current_size + 1, self.buffer_size)

    def sample(self, batch_size, device=None, dtype=torch.float32):
        """Sample a batch from buffer.
        For an 'obs', if its 'trun' is True, it will not be sampled; 
        If its 'term' is True, it will be sampled.
        For example, if the buffer is like:
            obs:  o1, o2, o3, o4, o5, o6, o7, o8, o9
            term: F,  F,  F,  F,  F,  F,  T,  F,  F
            trun: F,  F,  F,  T,  F,  F,  F,  F,  F
        Then the transition1 '(o4,a4,r4,o5,d5)' can not be sampled;
        But the transition2 '(o7,a7,r7,o8,d7)' can be sampled;
        Because although target-q is 0 when training with these two transitions, 
        the reward of the 'trun' is not the 'term' reward, which may lead to inaccurate estimation.
        """
        not_trun_index = np.where(~self._buffer['trun'][:self._current_size-1])[0]
        indexes = np.random.choice(not_trun_index, size=batch_size, replace=True)
        samples = _get_trans(self._buffer, indexes)
        samples['next_obs'] = _get_trans(self._buffer, indexes+1)['obs']
        if device is not None:
            _to_tensor(samples, device, dtype=dtype)
        return samples

    def init_offline(self, offlie_data, data_num, buffer_size=None):
        """Initiate the ReplayBuffer with offline data.
        If buffer_size is None, get all data in offlie_data. 
        If buffer_size < dataset's size, get the offlie_data[:buffer_size].
        If buffer_size > dataset's size, the offlie_data will be padded with zeros.
        For the data: 'obs': [a, b, c, d, e, f]
            buffer_size=None will return 'obs': [a, b, c, d, e, f]
            buffer_size=3 will return 'obs': [a, b, c]
            buffer_size=7 will return 'obs': [a, b, c, d, e, f, 0, 0]
        """
        self._buffer = offlie_data
        self._current_size = data_num
        self._pointer = 0
        self.buffer_size = data_num
        
        if buffer_size is not None:
            if buffer_size <= data_num:
                self._buffer = _get_trans(self._buffer, list(range(buffer_size)))
                self._current_size = buffer_size
                self._pointer = 0
            else:
                _pad_buffer(self._buffer, pad_size = buffer_size-data_num)
                self.episode_steps = np.concatenate((self.episode_steps, np.zeros(buffer_size-data_num, dtype=np.int32)))
                self._current_size = data_num
                self._pointer = data_num
            self.buffer_size = buffer_size
        
        print(f"The offline data num is: {data_num}, and the buffer_size is: {self.buffer_size}.")
        
    def save(self):
        # save to hdf5
        pass

    def load(self):
        # load from hdf5
        pass

class TrajectoryBuffer(BaseBuffer):
    def __init__(self, buffer_size: int = 0):
        super().__init__(buffer_size)

    def add(self, transition: Dict, step: int):
        assert (self._pointer < self.buffer_size
                ), 'The buffer is full now. It needs to be "clear" before next add.'
        if not self._buffer:
            _build_buffer(self._buffer, transition, self.buffer_size)

        _add_tran(self._buffer, transition, index=self._pointer)
        self.episode_steps[self._pointer] = step

        self._pointer += 1
        self._current_size = self._pointer
    
    def sample(self, batch_size, device=None, dtype=torch.float32):
        assert (self._pointer == self.buffer_size
                ), f'The buffer is not full. The expected size is {self.buffer_size}, but now is {self._pointer}'
        if batch_size == self.buffer_size:
            samples = self._buffer
        else:
            indexes = np.random.choice(self._pointer-1, size=batch_size, replace=True)
            samples = _get_trans(self._buffer, indexes)
        if device is not None:
            _to_tensor(samples, device, dtype=dtype)
        return samples
    
    def finish_path(self, agent):
        """This method is called at the end of a trajectory.
        The 'values', 'gae_advs' and 'log_probs' will be computed in this function.
        """
        obs = torch.as_tensor(self._buffer['obs'], dtype=torch.float32, device=agent.device)
        act = torch.as_tensor(self._buffer['act'], dtype=torch.float32, device=agent.device)
        rew = torch.as_tensor(self._buffer['rew'], dtype=torch.float32, device=agent.device)
        next_obs = torch.as_tensor(self._buffer['next_obs'], dtype=torch.float32, device=agent.device)
        done = torch.as_tensor(self._buffer['done'], dtype=torch.float32, device=agent.device)

        gamma = agent.gamma
        gae_lambda = agent.gae_lambda
        gae_normalize = agent.gae_normalize

        with torch.no_grad():
            values = agent.critic(obs).squeeze()
            next_values = agent.critic(next_obs).squeeze()
            log_probs, _ = agent.actor.get_logprob_entropy(obs, act)

        gae = 0
        gae_advs = torch.zeros_like(rew, device=agent.device)
        for i in reversed(range(len(rew))):
            delta = rew[i] + gamma * next_values[i] * (1 - done[i]) - values[i]
            gae = delta + gamma * gae_lambda * gae * (1 - done[i])
            gae_advs[i] = gae

        if gae_normalize:
            gae_advs = (gae_advs - torch.mean(gae_advs) / torch.std(gae_advs))

        self._buffer['values'] = values.cpu().numpy()
        self._buffer['log_probs'] = log_probs.cpu().numpy()
        self._buffer['gae_advs'] = gae_advs.cpu().numpy()


    def clear(self):
        self._buffer.clear()
        self._pointer = 0
    
    def save(self):
        # save to hdf5
        pass

    def load(self):
        # load from hdf5
        pass

class VectorBuffer:
    def __init__(self, 
                 buffer_size: int, # The size of total buffer.
                 buffer_num: int,  # The number of buffer in vector buffers.
                 buffer_class: BaseBuffer):
        self.buffer_size = buffer_size
        self.buffer_num = buffer_num
        self.per_buffer_size = int(self.buffer_size / self.buffer_num)  # The size of each buffer in VectorBuffer.
        self.buffer_list = [buffer_class(self.per_buffer_size) for _ in range(buffer_num)]

    def add(self, transitions: List[Dict], steps: List[int]) -> None:
        assert (self.buffer_num == len(transitions)
                ), "The num of envs is not equal to the num of buffers."
        for i in range(self.buffer_num):
            self.buffer_list[i].add(transitions[i], steps[i])
        self.current_size = sum([self.buffer_list[i]._current_size for i in range(self.buffer_num)])

    def sample(self, batch_size, device=None, dtype=torch.float32) -> Dict:
        buffer_indx = np.random.choice(self.buffer_num, batch_size, replace=True)
        sample_nums = np.bincount(buffer_indx)
        sample_list = []
        for buffer_id, sample_num in enumerate(sample_nums):
            sample_list.append(self.buffer_list[buffer_id].sample(sample_num))
        samples = _concat_batch(sample_list)
        if device is not None:
            _to_tensor(samples, device, dtype=dtype)
        return samples

    def save(self):
        pass

    def load(self):
        pass

    def finish_path(self, agent):
        for buffer in self.buffer_list:
            buffer.finish_path(agent)

    def clear(self):
        for buffer in self.buffer_list:
            buffer.clear()
