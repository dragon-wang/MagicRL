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
                     "visual_0": np.ndarray (b2, c2)
                     ...
    "act": "discrete": np.ndarray ()  # scalar
           "continuous": np.ndarray (e1, )
    "rew": np.ndarray ()  # scalar
    "done": np.ndarray ()  # scalar

    If buffer_size = n, the buffer is initialized like:
    "obs": "vector": "vector_0": np.empty (n, a1)
                     "vector_1": np.empty (n, a2)
                     ...
           "visual": "visual_0": np.empty (n, b1, c1)
                     "visual_0": np.empty (n, b2, c2)
                     ...
    "act": "discrete": np.empty (n, )
           "continuous": np.empty (n, e1)
    "rew": np.empty (n, )  # scalar
    "done": np.empty (n, )  # scalar

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
    "obs": [a, 0, 0, 0, 0]
    "act": [b, 0, 0, 0, 0]
    "rew": [c, 0, 0, 0, 0]
    "done": [d, 0, 0, 0, 0]

    And the transition is:
    "obs": c
    "act": d
    "rew": e
    "done": f

    After call _add_tran(buffer, transition, 1), the buffer is:
    "obs": [a, 1, 0, 0, 0]
    "act": [b, 1, 0, 0, 0]
    "rew": [c, 1, 0, 0, 0]
    "done": [d, 1, 0, 0, 0]

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
    """Get one or several transitions from the buffer according to the index
    If index is i, the result will squeeze the dim;
    If index is [i] or [i,j], the result will not squeeze the dim;

    If the buffer with buffer_size = 5 is:
    "obs": [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
    "act": [[d1, d2, d3], [e1, e2, e3], [f1, f2, f3]]

    The result of _get_trans(buffer, 1) is:
    "obs": [b1, b2, b3]
    "act": [e1, e2, e3]

    The result of _get_trans(buffer, [1]) is:
    "obs": [[b1, b2, b3]]
    "act": [[e1, e2, e3]]

    The result of _get_trans(buffer, [1,2]) is:
    "obs": [[a2, a3], [b2, b3], [c2, c3]]
    "act": [[d2, d3], [e2, e3], [f2, f3]] 

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

def _concat_batch(batch_list: Union[List[Dict], Tuple[Dict]]):
    batches = {}
    for k, v in batch_list[0].items():
        if not isinstance(v, Dict):
            batches[k] = np.concatenate([batch_list[i][k] for i in range(len(batch_list))])
        else: 
            batches[k] = _concat_batch([batch_list[i][k] for i in range(len(batch_list))])
    return batches

class BaseBuffer(ABC):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
    
    @abstractmethod
    def add(self, transition: Dict):
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
    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)
        
        self._buffer = {}
        self._pointer = 0  # Point to the current position in the buffer
        self._current_size = 0  # The current size of the buffer

    def add(self, transition: Dict):
        if not self._buffer:
            _build_buffer(self._buffer, transition, self.buffer_size)
        _add_tran(self._buffer, transition, index=self._pointer)
        self._pointer = (self._pointer + 1) % self.buffer_size
        self._current_size = min(self._current_size + 1, self.buffer_size)

    def sample(self, batch_size, device=None, dtype=torch.float32):
        indexes = np.random.choice(self._current_size, size=batch_size, replace=True)
        samples = _get_trans(self._buffer, indexes)
        if device is not None:
            _to_tensor(samples, device, dtype=dtype)
        return samples

    def save(self):
        # save to hdf5
        pass

    def load(self):
        # load from hdf5
        pass

class VectorBuffer(BaseBuffer):
    def __init__(self, buffer_size: int, buffer_num: int, buffer_class: BaseBuffer):
        super().__init__(buffer_size)  # The size of total buffer.
        self.buffer_num = buffer_num
        self.per_buffer_size = int(self.buffer_size / self.buffer_num)  # The size of each buffer in VectorBuffer.
        self.buffer_list = [buffer_class(self.per_buffer_size) for _ in range(buffer_num)]

    def add(self, transitions: List[Dict]) -> None:
        assert (self.buffer_num == len(transitions)
                ), "The num of envs is not equal to the num of buffers."
        for i in range(self.buffer_num):
            self.buffer_list[i].add(transitions[i])
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
