from abc import abstractmethod
import numpy as np
import pandas as pd
import gymnasium as gym
from tianshou.env import ShmemVectorEnv, DummyVectorEnv
from collections import OrderedDict
from numpy import ndarray
import torch
import tianshou
from typing import Union, Optional, Sequence, List, Tuple, Dict, Any
from gymnasium import spaces


class BaseSimulator:
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, random_init: bool) -> OrderedDict[str, ndarray]:
        """
        actiuvate the simulator and return the init state
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, action: Union[dict, float], state: dict, integral_num: int = 10) -> OrderedDict[str, ndarray]:
        """
        update the model from certain state with given action
        action is the agent's choosen action
        state is the t-1 state
        integral_num is the number of integral in a unit time (i.e. 1/dt)
        """
        raise NotImplementedError


class BaseReward:
    def __init__(self):
        pass

    @abstractmethod
    def count_reward(self, *args, **kwargs) -> float:
        """
        actiuvate the simulator and return the init state
        """
        raise NotImplementedError


class EpisodicEnv(gym.Env):
    """
    receive a fixed time series window and give outcomes. The episode length is always the same.
    """

    def __init__(self, ep_len, generator: torch.nn.Module, mort_predictor: torch.nn.Module,
                 patient_first_obs: np.array, action_encode_fn, safe_step=5, device="cpu"):
        self.ep_len = ep_len
        self.safe_step = safe_step  # the first safe_step steps are not terminated

        self.generator = generator.to(device)
        self.generator.eval()
        self.mort_predictor = mort_predictor.to(device)
        self.mort_predictor.eval()

        self.patient_first_obs = patient_first_obs
        self.patient_list = list(range(patient_first_obs.shape[0]))

        self.action_encoder = action_encode_fn
        self.generator_state = None
        self.mort_predictor_state = None

        self.cur_obs = None
        self.device = device

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            option: Optional[dict] = None,
    ):
        self.generator_state = None
        self.mort_predictor_state = None
        self._step = 0

        if option is None or "patient_id" not in option.keys() or option["patient_id"] is None:
            patient_id = np.random.choice(self.patient_list)
        else:
            patient_id = option["patient_id"]
            assert patient_id in self.patient_list

        self.terminated = False
        self.truncated = False
        self.cur_obs = self.patient_first_obs[patient_id]
        info = {"step": self._step}
        return self.cur_obs, info

    def step(self, action):
        action = int(action)

        done, mort_risk = self.get_outcome(self.cur_obs, action)

        self._step += 1
        obs_next = self.get_next_obs(action)
        self.cur_obs = obs_next

        if done:
            self.terminated = True
        if self._step >= self.ep_len and not self.terminated:
            self.truncated = True

        # get reward
        interm_r = self.get_interm_reward(self.cur_obs, action)
        final_r = self.get_final_reward(done) if done else 0
        final_r = final_r * done
        r = interm_r + final_r

        return obs_next.cpu().detach().numpy(), r, self.truncated, self.terminated, {
            "mort_risk": mort_risk, "outcome": done, "step": self._step}

    def seed(self, seed):
        super().reset(seed=seed)
        np.random.seed(seed)

    def get_next_obs(self, action):

        z = self.generator.sample_Z(1, 1, self.generator.obs_len)
        obs_next, self.generator_state = self.generator(z, action, self.generator_state)
        return obs_next

    def get_interm_reward(self, obs, action):
        return 0

    def get_final_reward(self, done):
        # done means death
        r = -100 if done else 100
        return r

    def get_outcome(self, obs, action):
        # 0: alive, 1: death

        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).to(self.device)
        if not torch.is_tensor(action):
            action = torch.tensor(action).to(self.device)
        x = torch.cat([obs, action], dim=-1)
        mort_risk, self.mort_predictor_state = self.mort_predictor(x, state=self.mort_predictor_state)
        mort_risk = mort_risk.item()
        outcome = np.random.choice([0, 1], p=[1 - mort_risk, mort_risk])
        return outcome, mort_risk


def make_env(task, seed, training_num, test_num, num_actions, **kwargs):
    try:
        import envpool
    except:
        # warnings.warn("envpool not installed, switch to for loop")
        envpool = None
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(
            task, num_envs=training_num, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        env = gym.make(task, n_act=num_actions)
        if training_num > 1:
            train_envs = ShmemVectorEnv(
                [lambda: gym.make(task, n_act=num_actions) for _ in range(training_num)]
            )
            test_envs = ShmemVectorEnv([lambda: gym.make(task, n_act=num_actions) for _ in range(test_num)])

        else:
            train_envs = DummyVectorEnv([lambda: gym.make(task, n_act=num_actions)])
            test_envs = DummyVectorEnv([lambda: gym.make(task, n_act=num_actions)])
        env.unwrapped.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    return env, train_envs, test_envs


class DiabetesRiskReward(BaseReward):
    def __init__(self):
        super().__init__()

    def risk_fn(self, BG: float):
        f = 1.509 * (np.log(BG) ** 1.084 - 5.381)
        risk = 10 * np.power(f, 2)
        return risk

    def count_reward(self, BG, risk_prev):
        BG = float(BG)
        risk_prev = float(risk_prev)
        risk_cur = self.risk_fn(BG)
        reward = risk_prev - risk_cur
        return reward, risk_cur


def uniform_random(mean, width, absolute=False):
    def single_random(mean, width):
        if absolute:
            return float(np.random.uniform(mean - width, mean + width))
        else:
            return float(np.random.uniform(mean - mean * width, mean + mean * width))

    """
    :return: a random number in [mean-width, mean+width] with uniform distribution
    """
    if isinstance(mean, (list, pd.core.series.Series, np.ndarray)):
        out = []
        for m in mean:
            out.append(single_random(m, width))
        return out
    elif isinstance(mean, (int, float, np.float32, np.float64, np.int64, np.int32)):
        return single_random(mean, width)
    else:
        raise NotImplementedError(f"mean type {type(mean)} not supported")
