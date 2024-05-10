from typing import Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import ndarray
from typing import Any, SupportsFloat, TypeVar
from collections import OrderedDict
from scipy.integrate import solve_ivp

from DTRGym.base import BaseSimulator, BaseReward, uniform_random
from DTRGym.utils import DiscreteActionWrapper

ObsType = TypeVar("ObsType")

"""
Env used in the paper 'Drug Scheduling of Cancer Chemotherapy based on Natural Actor-critic Approach'
    and 'Reinforcem net Learning-based control of drug dosing for cancer chemotherapy treatment'

State:
    {N, T, I, B}
    N refers to the number of normal cell
    T refers to the number of tumour cell
    I refers to the number of immune cell
    B refers to the drug concentration in the blood system

Action:
    A float between 0 and 1. 1 refers to the maximum dose level, vice versa.

Reward
    A float. 
"""


class AhnReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state, init_state, action, terminated) -> float:
        N, T, I, B = state["N"], state["T"], state["I"], state["B"]
        N0, T0, I0, B0 = init_state["N"], init_state["T"], init_state["I"], init_state["B"]
        reward = N/N0 - T/T0 - action
        if terminated:  # patient die
            return -100
        return float(reward)


class AhnODE(BaseSimulator):
    def __init__(self, state_noise, pkpd_noise):
        super().__init__()
        """
        Time unit is 1 day
        State:
            N refers to the number of normal cell
            T refers to the number of tumour cell
            I refers to the number of immune cell
            B refers to the drug concentration in the blood system
        """
        self.state_noise = state_noise
        self.pkpd_noise = pkpd_noise
        self.cur_time = 0  # unit is day
        self.time_interv = 0.25  # 6h

    def activate(self) -> OrderedDict[str, ndarray]:
        """
        The default init state is the case 2 in the paper 'Drug scheduling of cancer chemotherapy ...'
        as its simulation results looks more plausible.
        params:
            pkpd_noise: float: the noise level of pharmacokinetic and pharmacodynamic
        """
        width = self.pkpd_noise * 0.5
        self.r2, self.b2, self.c4, self.a3 = uniform_random(1., width), uniform_random(1., width), uniform_random(
            1., width), uniform_random(0.1, width)
        self.r1, self.b1, self.c2, self.c3, self.a2 = uniform_random(1.5, width), uniform_random(1., width), \
            uniform_random(0.5, width), uniform_random(1., width), uniform_random(0.3, width)
        self.s, self.rho, self.alpha, self.c1, self.d1, self.a1 = uniform_random(0.33, width), uniform_random(0.01,
                                                                                                              width), \
            uniform_random(0.3, width), uniform_random(1., width), uniform_random(0.2, width), uniform_random(0.2,
                                                                                                              width)
        self.d2 = uniform_random(1., width)

        init_state = OrderedDict({"N": np.array([uniform_random(0.9, self.state_noise*0.5)], dtype=np.float32),
                                  "T": np.array([uniform_random(0.2, self.state_noise*0.5)], dtype=np.float32),
                                  "I": np.array([uniform_random(0.005, self.state_noise*0.5)], dtype=np.float32),
                                  "B": np.array([0.0], dtype=np.float32)})
        return init_state

    def update(self, action: Union[dict, float], state: OrderedDict) -> OrderedDict[str, Union[float, Any]]:
        # update step is 0.25 day

        def odes_fn(t, variables, u):
            N, T, I, B = variables
            dNdt = self.r2 * N * (1 - self.b2 * N) - self.c4 * T * N - self.a3 * (1 - np.exp(-B)) * N
            dTdt = self.r1 * T * (1 - self.b1 * T) - self.c2 * I * T - self.c3 * T * N - self.a2 * (1 - np.exp(-B)) * T
            dIdt = self.s + self.rho * I * T / (self.alpha + T) - self.c1 * I * T - self.d1 * I - self.a1 * (
                        1 - np.exp(-B)) * I
            dBdt = -self.d2 * B + u

            if self.state_noise > 0:
                noise = np.random.normal(0, self.state_noise, 4)
                dNdt += dNdt * noise[0]
                dTdt += dTdt * noise[1]
                dIdt += dIdt * noise[2]
                dBdt += dBdt * noise[3]
            return [dNdt, dTdt, dIdt, dBdt]

        variables = np.array((state["N"], state["T"], state["I"], state["B"])).flatten()

        # Solve the ODEs with a fixed B value for the current time interval
        solution = solve_ivp(odes_fn, (self.cur_time, self.cur_time + self.time_interv),
                             variables, args=(action[0],))

        N, T, I, B = solution.y[0, -1], solution.y[1, -1], solution.y[2, -1], solution.y[3, -1]
        N, T, I, B = max(0, N), max(0, T), max(0, I), max(0, B)
        self.cur_time += self.time_interv
        return OrderedDict({"N": np.array([N], dtype=np.float32),
                            "T": np.array([T], dtype=np.float32),
                            "I": np.array([I], dtype=np.float32),
                            "B": np.array([B], dtype=np.float32)})


class AhnChemoEnv(gym.Env):
    def __init__(self, max_t: int = 600, delayed_steps=0,
                 obs_noise=0.2, state_noise=0.5, pkpd_noise=0.1,
                 missing_rate=0.0, **kwargs):
        """
        :param max_t: each step is 6hours, so max_t=600 means 150 days
        :param delayed_rew: give reward at the end of each *STEP*. *STEP* is 0.25 day

        State space: N: Normal Cell Population, T: Tumor Cell Population, I: Immune Cell Polulation, B: Drug Concentration
        Observation space: T: Tumor Cell Population, I: Immune Cell Polulation, B: Drug Concentration
        """
        super().__init__()
        self.Simulator = AhnODE(state_noise=state_noise, pkpd_noise=pkpd_noise)
        self.Reward = AhnReward()
        self.obs_noise = obs_noise
        self.state_noise = state_noise
        self.pkpd_noise = pkpd_noise
        self.max_t = max_t
        self.missing_rate = missing_rate

        self.init_state = None
        self.cur_state = None
        self.last_obs = None
        self.terminated = False
        self.truncated = False
        self.t = None
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                            high=np.array([2., 2.0, 1.0]), shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.env_info = {'action_type': 'continuous', 'reward_range': (0, 4),
                         "state_key": ["Normal Cell Population",
                                       "Tumor Cell Population",
                                       "Immune Cell Population",
                                       "Drug Concentration"],
                         "obs_key": ["Tumor Cell Population",
                                     "Immune Cell Population",
                                     "Drug Concentration"],
                         "act_key": ["Drug Dosage"]}
        self.state_map = {"N": "Normal Cell Population",
                          "T": "Tumor Cell Population",
                          "I": "Immune Cell Population",
                          "B": "Drug Concentration"}
        self.delayed_steps = delayed_steps
        if self.delayed_steps > 0:
            self.acc_rew = 0  # accumulated reward by day

    def _state2obs(self, state: OrderedDict, enable_missing: bool) -> np.ndarray:
        obs = np.array([state['T'], state['I'], state["B"]]).flatten()  # consider normal cell count as hidden state
        obs_noise = self.obs_noise * obs * np.random.uniform(-0.5, 0.5, size=obs.shape)
        obs = obs + obs_noise
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high).astype(np.float32)

        if enable_missing and np.random.uniform(0, 1) < self.missing_rate:
            obs = self.last_obs
        else:
            self.last_obs = obs
        return obs

    def reset(self, seed: int = None, **kwargs) -> Tuple[ObsType, dict[str, Any]]:
        self.seed(seed)
        self.t = 0
        self.terminated = False
        self.truncated = False
        # get init state
        init_state = self.Simulator.activate()
        self.init_state = init_state
        self.cur_state = init_state
        init_observation = self._state2obs(init_state, enable_missing=False)
        if self.delayed_steps > 0:
            self.acc_rew = 0  # accumulated reward by day
        init_state = {self.state_map[key]: value[0] for key, value in init_state.items()}
        info = {"state": init_state, "action": np.zeros(shape=(1,)), "instantaneous_reward": 0}

        self.spec.reward_threshold = self.env_info['reward_range'][1]

        return init_observation, info

    def seed(self, seed):
        super().reset(seed=seed)
        np.random.seed(seed)

    def step(self, action: float) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.terminated or self.truncated:
            raise RuntimeError("This treat is end, please call reset or export")
        if action < self.action_space.low or action > self.action_space.high:
            raise ValueError(f"action {action} should be in [{self.action_space.low}, {self.action_space.high}]")
        state_next = self.Simulator.update(action=action, state=self.cur_state)
        observation_next = self._state2obs(state_next, enable_missing=True)

        # check termination
        if state_next["N"] < self.init_state["N"] * 0.7:  # die
            self.terminated = True
            self.truncated = False
        if self.t + 1 == self.max_t:
            self.terminated = False
            self.truncated = True

        assert not (self.terminated and self.truncated)

        r = self.Reward.count_reward(
            state=self.cur_state, init_state=self.init_state, action=action, terminated=self.terminated)

        if self.delayed_steps > 0:
            self.acc_rew += r
            if self.t % self.delayed_steps == 0 or self.truncated or self.terminated:  # give reward every 2 days
                reward = self.acc_rew
                self.acc_rew = 0
            else:
                reward = 0
        else:
            reward = r

        self.t += 1
        self.cur_state = state_next
        state_next = {self.state_map[key]: value[0] for key, value in state_next.items()}
        info = {"state": state_next, "action": action, "instantaneous_reward": r}
        return observation_next, reward, self.terminated, self.truncated, info


def create_AhnChemoEnv_continuous(max_t: int = 600, n_act: int = 5, **kwargs):
    env = AhnChemoEnv(max_t, **kwargs)
    return env


def create_AhnChemoEnv_discrete(max_t: int = 600, n_act: int = 5, **kwargs):
    env = AhnChemoEnv(max_t, **kwargs)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_discrete_setting1(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.0, state_noise=0.0,
                      pkpd_noise=0.0, missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_discrete_setting2(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.0, state_noise=0.0,
                      pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_discrete_setting3(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.2, state_noise=0.1,
                      pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_discrete_setting4(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.5, state_noise=0.2,
                      pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_discrete_setting5(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.5, state_noise=0.2,
                      pkpd_noise=0.1, missing_rate=0.75)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_AhnChemoEnv_continuous_setting1(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.0, state_noise=0.0,
                      pkpd_noise=0.0, missing_rate=0.0)
    return env


def create_AhnChemoEnv_continuous_setting2(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.0, state_noise=0.0,
                      pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_AhnChemoEnv_continuous_setting3(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.2, state_noise=0.1,
                      pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_AhnChemoEnv_continuous_setting4(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.5, state_noise=0.2,
                      pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_AhnChemoEnv_continuous_setting5(max_t: int = 600, n_act: int = 5):
    env = AhnChemoEnv(max_t, delayed_steps=0, obs_noise=0.5, state_noise=0.2,
                      pkpd_noise=0.1, missing_rate=0.5)
    return env
