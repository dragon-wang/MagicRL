from typing import Any, Optional, Tuple, Union
from multiprocessing import Process, Pipe, connection

import gymnasium as gym
import numpy as np

from magicrl.env.utils import gymnasium_step_type
from magicrl.env.worker import BaseEnvWorker

def _worker(conn_parent: connection.Connection, conn_child: connection.Connection, env:gym.Env):
    conn_parent.close()
    try:
        while(True):
            cmd, data = conn_child.recv()
            if cmd == 'reset':
                reset_return = env.reset(**data)
                conn_child.send(reset_return)
            elif cmd == 'step':
                step_return = env.step(data)
                conn_child.send(step_return)
            elif cmd == 'render':
                render_return = env.render()
                conn_child.send(render_return)
            elif cmd == 'seed':
                env.reset(seed=data)
            elif cmd == 'close':
                close_return = env.close()
                conn_child.send(close_return)
                conn_child.close()
                break
            else:
                conn_child.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        conn_child.close()


class SubprocEnvWorker(BaseEnvWorker):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.conn_parent, self.conn_child = Pipe()
        self.process = Process(target=_worker, args=(self.conn_parent, self.conn_child, self.env), daemon=True)
        self.process.start()
        # self.pid = self.process.pid
        self.conn_child.close()

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.conn_parent.send(('reset', kwargs))
        else:
            self.conn_parent.send(('step', action))
    
    def recv(self) -> Union[Tuple[Any, dict], gymnasium_step_type]:
        result = self.conn_parent.recv()
        return result

    def seed(self, seed) -> None:
        self.action_space.seed(seed=seed)
        self.conn_parent.send(('seed', seed))

    def render(self) -> Any:
        self.conn_parent.send(('render', None))
        render_return = self.conn_parent.recv()
        return render_return

    def close_env(self)  -> Any:
        self.conn_parent.send(["close", None])
        close_return = self.conn_parent.recv()
        self.process.join()
        self.process.terminate()
        return close_return