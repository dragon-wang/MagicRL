from typing import Any, Tuple, Union
import numpy as np


gymnasium_step_type = Tuple[Any, np.ndarray, bool, bool, dict]
venvs_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]