import logging
from typing import Dict, List, Tuple

import numpy as np
from mlagents_envs.environment import (ActionTuple, DecisionSteps,
                                       TerminalSteps, UnityEnvironment)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel
from mlagents_envs.side_channel.side_channel import (IncomingMessage,
                                                     OutgoingMessage,
                                                     SideChannel)


class UnityWrapper:
    def __init__(self,
                 train_mode: bool = True,
                 env_path: str = None,
                 n_envs: int = 1,
                 base_port=5005,
                 worker_id=0,
                 no_graphics=True,
                 additional_args: List[str] = None,
                 seed=None):
        """
        Args:
            train_mode: If in train mode, Unity will run in the highest quality
            env_path: The executable path. The UnityEnvironment will run in editor if None
            n_envs: The env copies count
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            worker_id: Offset from base_port. Used for training multiple environments simultaneously.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
        """
        self.train_mode = train_mode
        self.env_path = env_path
        self.n_envs = n_envs
        self.base_port = base_port
        self.worker_id = worker_id
        self.no_graphics = no_graphics
        self.additional_args = additional_args
        self.seed = seed

        self._logger = logging.getLogger('UnityWrapper.Process')

        self.time_scale = 20 if train_mode else 1

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()
        self._env = UnityEnvironment(file_name=self.env_path,
                                     worker_id=worker_id,
                                     base_port=None,
                                     seed=seed,
                                     no_graphics=no_graphics,
                                     additional_args=self.additional_args,
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel])
        self.engine_configuration_channel.set_configuration_parameters(width=200 if train_mode else 1280,
                                                                       height=200 if train_mode else 720,
                                                                       quality_level=2,
                                                                       # 可在Unity的Edit → Project Settings → Quality中查看
                                                                       # 0: URP-Performant-Renderer
                                                                       # 1: URP-Balanced-Renderer
                                                                       # 2: URP-HighFidelity-Renderer
                                                                       time_scale=self.time_scale)
        self.environment_parameters_channel.set_float_parameter('env_copys', float(n_envs))
        
        # Initialize the Unity environment
        self.init()
        
    def init(self):
        """Initializes the Unity environment and retrieves observation and action specifications.
        """
        self.ma_obs_names: dict[str, list[str]] = {}
        self.ma_obs_shapes: dict[str, tuple[int, ...]] = {}
        self.ma_d_action_sizes: dict[str, list[int]] = {}
        self.ma_c_action_size: dict[str, int] = {}

        self._env.reset()
        self.behavior_names: list[str] = list(self._env.behavior_specs)

        for b_n in self.behavior_names:
            behavior_spec = self._env.behavior_specs[b_n]
            obs_names = [o.name for o in behavior_spec.observation_specs]
            self._logger.info(f'{b_n} Observation names: {obs_names}')
            self.ma_obs_names[b_n] = obs_names

            obs_shapes = [o.shape for o in behavior_spec.observation_specs]

            self._logger.info(f'{b_n} Observation shapes: {obs_shapes}')
            self.ma_obs_shapes[b_n] = obs_shapes

            self._empty_action = behavior_spec.action_spec.empty_action

            discrete_action_sizes = []
            if behavior_spec.action_spec.discrete_size > 0:
                for branch, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                    discrete_action_sizes.append(branch_size)
                    self._logger.info(f"{b_n} Discrete action branch {branch} has {branch_size} different actions")

            continuous_action_size = behavior_spec.action_spec.continuous_size

            self._logger.info(f'{b_n} Continuous action size: {continuous_action_size}')

            self.ma_d_action_sizes[b_n] = discrete_action_sizes  # list[int]
            self.ma_c_action_size[b_n] = continuous_action_size  # int

            for o_name, o_shape in zip(obs_names, obs_shapes):
                if ('camera' in o_name.lower() or 'visual' in o_name.lower() or 'image' in o_name.lower()) \
                        and len(o_shape) >= 3:
                    self.engine_configuration_channel.set_configuration_parameters(quality_level=2)
                    break

        self._logger.info('Initialized')

        return (self.ma_obs_names,
                self.ma_obs_shapes,
                self.ma_d_action_sizes,
                self.ma_c_action_size)
    
    def reset(self, reset_config: dict=None):
        """Resets the Unity environment and returns the initial observations and agent IDs.
        Args:
            reset_config (dict, optional): A dictionary used to set environment parameters, where the key (k) is the parameter name and the value (v) is the parameter value.
        Returns:
            ma_agent_ids: The agent ID corresponding to each behavior_name, for example:
                {'team=0': [0, 1, 2], 'team=1': [3, 4]}
            ma_obs_list: The observation list of agents corresponding to each behavior_name, with the order matching that of ma_agent_ids, for example:
                {
                    'team=0': [obs_0, obs_1, obs_2],
                    'team=1': [obs_3, obs_4]
                }
            In this case, obs_0 corresponds to the observation of the agent with agent_id 0, obs_1 corresponds to the observation of the agent with agent_id 1, and so on.
        """
        if reset_config:
            for k, v in reset_config.items():
                self.environment_parameters_channel.set_float_parameter(k, float(v))
        self._env.reset()
        ma_obs_list, ma_agent_ids = {}, {}
        
        for b_n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(b_n)
            ma_agent_ids[b_n] = decision_steps.agent_id
            ma_obs_list[b_n] = decision_steps.obs

        return ma_agent_ids, ma_obs_list
    
    def sample_random_actions(self):
        """Sample random actions for all agents in the Unity environment.
        Returns:
            c_actions: Sampled continuous actions.
            d_actions: Sampled discrete actions.
        """
        
        c_actions = {}
        d_actions = {}
        for b_n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(b_n)
            action_spec = self._env.behavior_specs[b_n].action_spec
            # Get sampled actions
            sampled_action = action_spec.random_action(n_agents=len(decision_steps))
            # Here, c_actions and d_actions are dicts, where the key is behavior_name and the value is the corresponding action
            c_actions[b_n] = sampled_action.continuous
            d_actions[b_n] = sampled_action.discrete

        return c_actions, d_actions

    def step(self, c_actions, d_actions):
        """Act on the Unity environment with the given actions.

        Args:
            c_actions (dict): The continuous actions to be executed, where the key (k) is the behavior_name, and the value (v) is the corresponding array of continuous actions.
            d_actions (dict): The discrete actions to be executed, where the key (k) is the `behavior_name`, and the value (v) is the corresponding array of discrete actions.

        Returns:
            decision_ma_agent_ids: The agent IDs in Decision_steps.
            decision_ma_obs_list: The list of agent observations in Decision_steps, ordered corresponding to decision_ma_agent_ids.
            decision_ma_last_reward: The rewards of agents in Decision_steps.
            terminal_ma_agent_ids: The agent IDs in Terminal_steps.
            terminal_ma_obs_list: The list of agent observations in Terminal_steps, ordered corresponding to terminal_ma_agent_ids.
            terminal_ma_last_reward: The rewards of agents in Terminal_steps.
            terminal_ma_max_reached: Whether agents in Terminal_steps are done due to reaching the maximum timestep.
        """

        # Set actions for all agents
        for b_n in self.behavior_names:
            self._env.set_actions(b_n,  ActionTuple(continuous=c_actions[b_n],
                                                    discrete=d_actions[b_n]))

        # Execute actions: It will run until Unity requests actions or some agents are done.
        # That is, the decision_steps and terminal_steps obtained below are not empty.
        # If step() is called without set_actions, it will have no effect on agents.
        self._env.step()

        decision_ma_agent_ids: dict[str, np.ndarray] = {}
        decision_ma_obs_list: dict[str, list[np.ndarray]] = {}
        decision_ma_last_reward: dict[str, np.ndarray] = {}

        terminal_ma_agent_ids: dict[str, list[int]] = {}
        terminal_ma_obs_list: dict[str, list[np.ndarray]] = {}
        terminal_ma_last_reward: dict[str, np.ndarray] = {}
        terminal_ma_max_reached: dict[str, np.ndarray] = {}

        # Obtain feedback from the environment after executing actions.
        for b_n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(b_n)
            
            decision_ma_agent_ids[b_n] = decision_steps.agent_id
            decision_ma_obs_list[b_n] = decision_steps.obs
            decision_ma_last_reward[b_n] = decision_steps.reward

            terminal_ma_agent_ids[b_n] = terminal_steps.agent_id
            terminal_ma_obs_list[b_n] = terminal_steps.obs
            terminal_ma_last_reward[b_n] = terminal_steps.reward
            terminal_ma_max_reached[b_n] = terminal_steps.interrupted
        
        return (
            decision_ma_agent_ids,
            decision_ma_obs_list,
            decision_ma_last_reward
        ), (
            terminal_ma_agent_ids,
            terminal_ma_obs_list,
            terminal_ma_last_reward,
            terminal_ma_max_reached
        )

    def close(self):
        self._env.close()
        self._logger.warning(f'Environment closed')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    env = UnityWrapper(train_mode=False,
                    #    env_path=r"xxx",
                       n_envs=2,
                       base_port=5004,
                       worker_id=0,
                       no_graphics=False,
                       seed=10)
    
    obs, info = env.reset()

    for i in range(100000):
        c_actions, d_actions = env.sample_random_actions()
        env.step(c_actions, d_actions)  

        if (i+1) % 300 == 0:
            env.reset()