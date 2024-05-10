from typing import Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import ndarray
from typing import Any, SupportsFloat, TypeVar
from collections import OrderedDict
from scipy.integrate import solve_ivp
import logging
from numba import njit

from .base import BaseSimulator, BaseReward
from DTRGym.utils import DiscreteDoubleActionWrapper
from .base import uniform_random

ObsType = TypeVar("ObsType")

"""
Env developed in the paper 'A mixed radiotherapy and chemotherapy model for treatment of cancer with metastasis'
https://onlinelibrary.wiley.com/doi/full/10.1002/mma.3887?casa_token=55hZuYxUe0IAAAAA%3A0zfeXpqcZ01aioyDO8W8E9b79HiM3bhznRJbhJ7ssm94vtjh9nO2xICPh7RvRb1eF0acdom-BEfGBXw

Different from the original one, the dynamics of c1 and c2 are removed since there is unclear parameters in the corresponding ODEs.
c1 and c2 are treated as constant in this env.
Observation:
    {T_p, N_p, L_p, C, T_s, N_s, L_s, M}
    T_p: The total tumor cell population at the primary site
    N_p: The concentration of NK cells liter of blood (cells/L) at the primary site
    L_p: The concentration of CD8+T cells liter of blood (cells/L) at the primary site
    C: The concentration of lymphocytes liter of blood (cells/L)
    T_s: The total tumor cell population at the secondary site
    N_s: The concentration of NK cells liter of blood (cells/L) at the secondary site
    L_s: The concentration of CD8+T cells liter of blood (cells/L) at the secondary site
    M: The concentration of chemotherapy agent in the blood (mg/L)
Action:
    {D, v}
    D: the effect of radiotherapy
    v: the effect of chemotherapy

Reward
    A float. 
"""


class GhaffariReward(BaseReward):
    def __init__(self):
        super().__init__()

    def count_reward(self, state, init_state, action,
                     positive_terminated, negative_terminated) -> float:
        T_p, T_s, N_p, N_s = state["T_p"], state["T_s"], state["N_p"], state["N_s"]
        T_p0, T_s0, N_p0, N_s0 = init_state["T_p"], init_state["T_s"], init_state["N_p"], init_state["N_s"]
        T0 = T_p0 + T_s0
        T = T_p + T_s
        N0 = N_p0 + N_s0
        N = N_p + N_s

        T, T0, N, N0 = max(np.e, T), max(np.e, T0), max(np.e, N), max(np.e, N0)
        T, T0, N, N0 = np.log(T), np.log(T0), np.log(N), np.log(N0)

        tumor_reduction = 1 - T / T0  # positive if T is less, negative if T is more
        # nk_penalty = -np.abs(N / N0 - 1)

        reward = tumor_reduction

        if positive_terminated:
            terminated_reward = 100
        elif negative_terminated:
            terminated_reward = -100
        else:
            terminated_reward = 0

        return float(reward + terminated_reward)


@njit
def odes_fn(t, variables, D, v_M, hparams):
    variables[variables < 0] = 0
    T_p, N_p, L_p, C, T_s, N_s, L_s, M, u, v, x = variables
    tau, delta, a1, b1, c1, d1, l, s, e1, f1, p1, m1, j1, k1, q1, r11, r12, u1, K_1T, K_1L, K_1N, K_1C, alpha, beta, mu, \
    a2, b2, c2, d2, e2, f2, p2, m2, j2, k2, q2, r21, r22, u2, K_2T, K_2L, K_2N, gamma1, gamma2, gamma3, epsilon, alpha1, \
    alpha2, W_1T, W_1N, W_1L, W_1C, W_2T, W_2N, W_2L = hparams

    D_p_denominator = max(1, s * T_p ** l + L_p ** l)
    D_p = d1*(L_p**l/D_p_denominator)
    dT_pdt = a1*T_p*(1-b1*T_p) - c1*N_p*T_p - D_p*T_p - alpha1*T_p - D*T_p + \
             gamma1*u - K_1T*(T_p*M/(W_1T+T_p))
    dN_pdt = e1*C - p1*N_p*T_p - f1*N_p - epsilon*D*N_p + gamma2*v - \
             K_1N*(N_p*M/(W_1N+N_p))
    dL_pdt = -m1*L_p + j1*(T_p/(k1+T_p))*L_p - q1*L_p*T_p + r11*N_p*T_p + \
             r12*C*T_p - u1*N_p*L_p**2\
             - epsilon*D*L_p + gamma3*x - K_1L*(L_p*M/(W_1L+L_p))

    dCdt = alpha - beta*C - K_1C*(C*M/(W_1C+C))

    D_s_denominator = max(1, s * T_s ** l + L_s ** l)
    D_s = d2*(L_s**l/D_s_denominator)

    if t > tau:
        dT_sdt = a2 * T_s * (1 - b2 * T_s) - c2 * N_s * T_s - D_s * T_s + alpha2 * \
                 T_p*(t-tau) - K_2T * (T_s * M / (W_2T + T_s))
        dN_sdt = e2 * C - p2 * N_s * T_s - f2 * N_s - K_2N * (N_s * M / (W_2N + N_s))
        dL_sdt = -m2 * L_s + j2 * (T_s / (k2 + T_s)) * L_s - q2 * L_s * T_s + \
                 r21 * N_s * T_s + r22 * C * T_s - u2 * N_s * L_s ** 2 - K_2L * \
                 (L_s * M / (W_2L + L_s))
    else:
        dT_sdt = 0
        dN_sdt = 0
        dL_sdt = 0

    dMdt = -mu*M + v_M
    dudt = D*T_p - gamma1*u - delta*u
    dvdt = D*N_p - gamma2*v - delta*v
    dxdt = D*L_p - gamma3*x - delta*x
    return [dT_pdt, dN_pdt, dL_pdt, dCdt, dT_sdt, dN_sdt, dL_sdt, dMdt, dudt, dvdt, dxdt]


class GhaffariODE(BaseSimulator):
    def __init__(self, pkpd_noise: float, state_noise: float):
        super().__init__()
        """
        Time unit is 1 day
        State:
            {T_p, N_p, L_p, C, T_s, N_s, L_s, c1, c2, M, u, v, x}
            T_p: The total tumor cell population at the primary site
            N_p: The concentration of NK cells liter of blood (cells/L) at the primary site
            L_p: The concentration of CD8+T cells liter of blood (cells/L) at the primary site
            C: The concentration of lymphocytes liter of blood (cells/L)
            T_s: The total tumor cell population at the secondary site
            N_s: The concentration of NK cells liter of blood (cells/L) at the secondary site
            L_s: The concentration of CD8+T cells liter of blood (cells/L) at the secondary site
            M: The concentration of chemotherapy agent in the blood (mg/L)
            u: The population of cancer cells that have been exposed to radiation
            v: The population of NK cells that have been exposed to radiation
            x: The population of CD8+T cells that have been exposed to radiation
        """
        assert 0 <= state_noise <= 1, "state_noise must be in [0, 1]"
        assert 0 <= pkpd_noise <= 1, "pkpd_noise must be in [0, 1]"

        self.pkpd_noise = pkpd_noise
        self.state_noise = state_noise

        self.cur_time = 0  # unit is day
        self.time_interv = 1

    def activate(self) -> OrderedDict[str, ndarray]:
        """
        The default init state is the case 2 in the paper 'Drug scheduling of cancer chemotherapy ...'
        as its simulation results looks more plausible.
        params:
            state_noise: the noise of initial state, must be in [0, 1]
        """
        self._init_ode_parameters(self.pkpd_noise)

        init_state = OrderedDict({"T_p": np.array([1e7], dtype=np.float32),
                                  "N_p": np.array([2e8], dtype=np.float32),
                                  "L_p": np.array([1e2], dtype=np.float32),
                                  "C": np.array([1.5e9], dtype=np.float32),
                                  "T_s": np.array([0], dtype=np.float32),
                                  "N_s": np.array([1e4], dtype=np.float32),
                                  "L_s": np.array([1e2], dtype=np.float32),
                                  "M": np.array([0], dtype=np.float32),
                                  "u": np.array([0], dtype=np.float32),
                                  "v": np.array([0], dtype=np.float32),
                                  "x": np.array([0], dtype=np.float32)})
        if self.state_noise > 0:
            for key, value in init_state.items():
                if key not in ("M", "u", "v", "x"):
                    v = uniform_random(value, self.state_noise*0.5)
                    init_state[key] = np.maximum(0, v).astype(np.float32)
            self.tau = np.random.randint(0, 10)  # delayed days for metastasis
        else:
            self.tau = 5
        self.cur_time = 0
        return init_state

    def _init_ode_parameters(self, pkpd_noise):
        param_dict = {"delta": 0.15, "a1": 4.31e-1, "b1": 1.02e-9, "c1": 6.41e-11, "d1": 2.34,
                      "l": 2.09, "s": 8.39e-2, "e1": 2.08e-7, "f1": 4.12e-2, "p1": 3.42e-6,
                      "m1": 2.04e-1, "j1": 2.49e-2, "k1": 3.66e7, "q1": 1.42e-6, "r11": 1.1e-7,
                      "r12": 6.5e-11, "u1": 3e-10, "K_1T": 100, "K_1L": 10, "K_1N": 10, "K_1C": 10,
                      "alpha": 7.5e8, "beta": 1.2e-2, "mu": 9e-1, "a2": 5, "b2": 1e-7,
                      "c2": 6.41e-12, "d2": 5, "e2": 2.08e-7, "f2": 3.5e-2, "p2": 1e-6,
                      "m2": 1.8e-1, "j2": 1.6e-2, "k2": 3.66e7, "q2": 1e-6, "r21": 2e-7, "r22": 7.5e-11,
                      "u2": 3e-10, "K_2T": 100, "K_2L": 10, "K_2N": 10, "K_2C": 10,
                      "gamma1": 0.04, "gamma2": 0.1, "gamma3": 0.1, "epsilon": 0.05, "alpha1": 1e-4, "alpha2": 1e-5,
                      "W_1T": 0.01, "W_1N": 1, "W_1L": 1, "W_1C": 1, "W_2T": 1, "W_2N": 1, "W_2L": 1}

        if pkpd_noise > 0:
            self.param_dict = {}
            width = pkpd_noise * 0.5
            for key in param_dict.keys():
                self.param_dict[key] = uniform_random(param_dict[key], width)
        else:
            self.param_dict = param_dict

        for key, value in self.param_dict.items():
            setattr(self, key, value)


    def update(self, action: Union[dict, float], state: OrderedDict) -> OrderedDict[str, Union[float, Any]]:
        variables = np.array([state["T_p"], state["N_p"], state["L_p"], state["C"],
                              state["T_s"], state["N_s"], state["L_s"], state["M"],
                              state["u"], state["v"], state["x"]], dtype=np.float32).flatten()

        D, v_M = action[0], action[1]
        hparams = np.array([self.tau, self.delta, self.a1, self.b1, self.c1, self.d1, self.l, self.s, self.e1, self.f1, self.p1,
                            self.m1, self.j1, self.k1, self.q1, self.r11, self.r12, self.u1, self.K_1T, self.K_1L, self.K_1N,
                            self.K_1C, self.alpha, self.beta, self.mu, self.a2, self.b2, self.c2, self.d2, self.e2, self.f2,
                            self.p2, self.m2, self.j2, self.k2, self.q2, self.r21, self.r22, self.u2, self.K_2T, self.K_2L,
                            self.K_2N, self.gamma1, self.gamma2, self.gamma3, self.epsilon, self.alpha1, self.alpha2,
                            self.W_1T, self.W_1N, self.W_1L, self.W_1C, self.W_2T, self.W_2N, self.W_2L])

        # Solve the ODEs with a fixed B value for the current time interval
        solution = solve_ivp(odes_fn, (self.cur_time, self.cur_time + self.time_interv),
                             variables, args=(D, v_M, hparams),
                             method='RK23',
                             atol=1e-4,
                             rtol=1e-2)
        T_p, N_p, L_p, C, T_s, N_s, L_s, M, u, v, x = solution.y[:, -1]

        self.cur_time += self.time_interv
        return OrderedDict({"T_p": np.array([T_p], dtype=np.float32),
                            "N_p": np.array([N_p], dtype=np.float32),
                            "L_p": np.array([L_p], dtype=np.float32),
                            "C": np.array([C], dtype=np.float32),
                            "T_s": np.array([T_s], dtype=np.float32),
                            "N_s": np.array([N_s], dtype=np.float32),
                            "L_s": np.array([L_s], dtype=np.float32),
                            "M": np.array([M], dtype=np.float32),
                            "u": np.array([u], dtype=np.float32),
                            "v": np.array([v], dtype=np.float32),
                            "x": np.array([x], dtype=np.float32)})


class GhaffariCancerEnv(gym.Env):
    def __init__(self, max_t: int = 200, delayed_steps=0,
                 obs_noise=0.2, state_noise=0.5, pkpd_noise=0.1,
                 missing_rate=0.0, **kwargs):
        """
        :param max_t: each step is 1 day, max_t=200 means 200 days
        :param delayed_rew: give reward at the end of every delayed_steps *STEP*. *STEP* is 1 day, by default a week

        Observation space: T: tumor population, I: immune population, B: drug concentration
        """
        super().__init__()
        self.Simulator = GhaffariODE(pkpd_noise=pkpd_noise, state_noise=state_noise)
        self.Reward = GhaffariReward()
        self.obs_noise = obs_noise
        self.state_noise = state_noise
        self.pkpd_noise = pkpd_noise
        self.missing_rate = missing_rate
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.max_t = max_t
        self.cur_state = None
        self.positive_terminated = False
        self.negative_terminated = False
        self.truncated = False
        self.t = None
        self.last_obs = None

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                            high=np.log([1e11, 1e10, 1e10, 1e11, 1e11, 1e10, 1e10]),
                                            shape=(7,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([10.0, 8.0]),
                                       shape=(2,),
                                       dtype=np.float32)
        self.state_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                      high=np.array([1e11, 1e10, 1e10, 1e11, 1e11, 1e10, 1e10]),
                                      shape=(7,),
                                      dtype=np.float32)
        self.env_info = {'action_type': 'continuous', 'reward_range': (-100, 1),
                         "state_key": ["T_p", "N_p", "L_p", "C", "T_s", "N_s", "L_s"],
                         "obs_key": ["T_p", "N_p", "L_p", "C", "T_s", "N_s", "L_s"],
                         "act_key": ["Radiation Dose (Gy)", "Chemotherapy Concentration"]}

        self.delayed_steps = delayed_steps
        self.init_state = None
        if self.delayed_steps > 0:
            self.acc_rew = 0  # accumulated reward by day

    def _state2obs(self, state: OrderedDict, enable_missing: bool) -> np.ndarray:
        obs = np.array([state['T_p'], state['N_p'], state['L_p'], state['C'],
                        state['T_s'], state['N_s'], state['L_s']]).flatten()

        obs_noise = self.obs_noise * obs * np.random.uniform(low=-0.5, high=0.5, size=obs.shape)
        obs = obs + obs_noise
        obs = np.where(obs < 1, 1, obs)
        obs = np.log(obs, dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        if enable_missing and np.random.uniform(0, 1) < self.missing_rate:
            obs = self.last_obs
        else:
            self.last_obs = obs
        return obs


    def reset(self, seed: int = None, **kwargs) -> Tuple[ObsType, dict[str, Any]]:
        self.seed(seed)
        self.t = 0
        self.positive_terminated = False
        self.negative_terminated = False
        self.truncated = False
        # get init state
        init_state = self.Simulator.activate()
        self.init_state = init_state
        self.cur_state = init_state
        init_observation = self._state2obs(init_state, enable_missing=False)
        if self.delayed_steps > 0:
            self.acc_rew = 0  # accumulated reward by day
        info = {"state": init_state, "action": np.zeros(shape=(2,)), "instantaneous_reward": 0}

        self.spec.reward_threshold = self.env_info['reward_range'][1]

        return init_observation, info

    def seed(self, seed):
        super().reset(seed=seed)
        np.random.seed(seed)

    def step(self, action: float) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        step the env with given action, and get next state and reward of this action
        """
        if self.positive_terminated or self.negative_terminated or self.truncated:
            raise RuntimeError("This treat is end, please call reset or export")
        # if action < self.action_space.low or action > self.action_space.high:
        #     raise ValueError(f"action should be in [{self.action_space.low}, {self.action_space.high}]")
        state_next = self.Simulator.update(action=action, state=self.cur_state)
        state_np = np.array([state_next['T_p'], state_next['N_p'], state_next['L_p'], state_next['C'],
                             state_next['T_s'], state_next['N_s'], state_next['L_s']], dtype=np.float32).flatten()
        observation_next = self._state2obs(state_next, enable_missing=True)
        
        # check termination
        if not self.state_space.contains(state_np):  # caused by oversize state.
            self.positive_terminated = False
            self.negative_terminated = True
            self.truncated = False
        if state_next["T_p"] < 1 and state_next["T_s"] < 1:  # no tumor
            self.positive_terminated = True
            self.negative_terminated = False
            self.truncated = False
        if self.t + 1 == self.max_t:
            self.positive_terminated = False
            self.negative_terminated = False
            self.truncated = True

        terminated = self.positive_terminated or self.negative_terminated

        r = self.Reward.count_reward(
            state=self.cur_state, init_state=self.init_state, action=action,
            positive_terminated=self.positive_terminated, negative_terminated=self.negative_terminated)
        if self.delayed_steps > 0:
            self.acc_rew += r
            if self.t % self.delayed_steps == 0 or self.truncated or terminated:  # give reward every 2 days
                reward = self.acc_rew
                self.acc_rew = 0
            else:
                reward = 0
        else:
            reward = r

        self.t += 1
        self.cur_state = state_next
        info = {"state": state_next, "action": action, "instantaneous_reward": r}
        return observation_next, float(reward), terminated, self.truncated, info


def create_GhaffariCancerEnv_discrete(max_t: int = 200, n_act: int = 25, **kwargs):
    env = GhaffariCancerEnv(max_t, **kwargs)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_continuous(max_t: int = 200, **kwargs):
    env = GhaffariCancerEnv(max_t, **kwargs)
    return env

def create_GhaffariCancerEnv_discrete_setting1(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.0, state_noise=0.0,
                            pkpd_noise=0.0, missing_rate=0.0)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_discrete_setting2(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.0, state_noise=0.0,
                            pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_discrete_setting3(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.1, state_noise=0.2,
                            pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_discrete_setting4(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.2, state_noise=0.5,
                            pkpd_noise=0.1, missing_rate=0.0)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_discrete_setting5(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.2, state_noise=0.5,
                            pkpd_noise=0.1, missing_rate=0.5)
    wrapped_env = DiscreteDoubleActionWrapper(env, n_act)
    return wrapped_env


def create_GhaffariCancerEnv_continuous_setting1(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.0, state_noise=0.0,
                            pkpd_noise=0.0, missing_rate=0.0)
    return env


def create_GhaffariCancerEnv_continuous_setting2(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.0, state_noise=0.0,
                            pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_GhaffariCancerEnv_continuous_setting3(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.1, state_noise=0.2,
                            pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_GhaffariCancerEnv_continuous_setting4(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.2, state_noise=0.5,
                            pkpd_noise=0.1, missing_rate=0.0)
    return env


def create_GhaffariCancerEnv_continuous_setting5(max_t: int = 200, n_act: int = 25):
    env = GhaffariCancerEnv(max_t, delayed_steps=0,
                            obs_noise=0.2, state_noise=0.5,
                            pkpd_noise=0.1, missing_rate=0.5)
    return env
