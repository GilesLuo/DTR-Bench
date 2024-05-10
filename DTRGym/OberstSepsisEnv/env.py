import warnings
from DTRGym.OberstSepsisEnv.state import State
from DTRGym.OberstSepsisEnv.action import Action
from DTRGym.OberstSepsisEnv.MDP import MDP
import gymnasium as gym
from typing import Union, Tuple, Dict, Any, Optional, List, Type, Callable, Sequence, cast
import numpy as np
from DTRGym.base import uniform_random
from gymnasium import spaces

"""
Includes blood glucose level proxy for diabetes: 0-3
    (lo2, lo1, normal, hi1, hi2); Any other than normal is "abnormal"
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics

Effect of vasopressors on if diabetic:
    raise blood pressure: normal -> hi w.p. .9, lo -> normal w.p. .5, lo -> hi w.p. .4
    raise blood glucose by 1 w.p. .5

Effect of vasopressors off if diabetic:
    blood pressure falls by 1 w.p. .05 instead of .1
    glucose does not fall - apply fluctuations below instead

Fluctuation in blood glucose levels (IV/insulin therapy are not possible actions):
    fluctuate w.p. .3 if diabetic
    fluctuate w.p. .1 if non-diabetic
Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4530321/

Additional fluctuation regardless of other changes
This order is applied:
    antibiotics, ventilation, vasopressors, fluctuations
"""

"""
Includes blood glucose level proxy for diabetes: 0-3
    (lo2 - counts as abnormal, lo1, normal, hi1, hi2 - counts as abnormal)
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics
"""


class MDP(object):
    def __init__(self, init_state_idx=None, init_state_idx_type='obs',
                 policy_array=None, policy_idx_type='obs', p_diabetes=0.2,
                 pkpd_noise=0.0):
        """
        initialize the simulator
        """
        assert p_diabetes >= 0 and p_diabetes <= 1, \
            "Invalid p_diabetes: {}".format(p_diabetes)
        assert policy_idx_type in ['obs', 'full', 'proj_obs']
        self.pkpd_noise = pkpd_noise

        # Check the policy dimensions (states x actions)
        if policy_array is not None:
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL
            if policy_idx_type == 'obs':
                assert policy_array.shape[0] == State.NUM_OBS_STATES
            elif policy_idx_type == 'full':
                assert policy_array.shape[0] == \
                       State.NUM_HID_STATES * State.NUM_OBS_STATES
            elif policy_idx_type == 'proj_obs':
                assert policy_array.shape[0] == State.NUM_PROJ_OBS_STATES

        # p_diabetes is used to generate random state if init_state is None
        self.p_diabetes = p_diabetes
        self.state = None

        # Only need to use init_state_idx_type if you are providing a state_idx!
        self.state = self.get_new_state(init_state_idx, init_state_idx_type)

        self.policy_array = policy_array
        self.policy_idx_type = policy_idx_type  # Used for mapping the policy to actions

    def get_new_state(self, state_idx=None, idx_type='obs', diabetic_idx=None):
        """
        use to start MDP over.  A few options:

        Full specification:
        1. Provide state_idx with idx_type = 'obs' + diabetic_idx
        2. Provide state_idx with idx_type = 'full', diabetic_idx is ignored
        3. Provide state_idx with idx_type = 'proj_obs' + diabetic_idx*

        * This option will set glucose to a normal level

        Random specification
        4. State_idx, no diabetic_idx: Latter will be generated
        5. No state_idx, no diabetic_idx:  Completely random
        6. No state_idx, diabetic_idx given:  Random conditional on diabetes
        """
        assert idx_type in ['obs', 'full', 'proj_obs']
        option = None
        if state_idx is not None:
            if idx_type == 'obs' and diabetic_idx is not None:
                option = 'spec_obs'
            elif idx_type == 'obs' and diabetic_idx is None:
                option = 'spec_obs_no_diab'
                diabetic_idx = np.random.binomial(1, self.p_diabetes)
            elif idx_type == 'full':
                option = 'spec_full'
            elif idx_type == 'proj_obs' and diabetic_idx is not None:
                option = 'spec_proj_obs'
        elif state_idx is None and diabetic_idx is None:
            option = 'random'
        elif state_idx is None and diabetic_idx is not None:
            option = 'random_cond_diab'

        assert option is not None, "Invalid specification of new state"

        if option in ['random', 'random_cond_diab']:
            init_state = self.generate_random_state(diabetic_idx)
            # Do not start in death or discharge state
            while init_state.check_absorbing_state():
                init_state = self.generate_random_state(diabetic_idx)
        else:
            # Note that diabetic_idx will be ignored if idx_type = 'full'
            init_state = State(
                state_idx=state_idx, idx_type=idx_type,
                diabetic_idx=diabetic_idx)

        return init_state

    def generate_random_state(self, diabetic_idx=None, state_idx=None):
        # Note that we will condition on diabetic idx if provided
        if diabetic_idx is None:
            diabetic_idx = np.random.binomial(1, self.p_diabetes)

        # hr and sys_bp w.p. [.25, .5, .25]
        hr_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        # percoxyg w.p. [.2, .8]
        percoxyg_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))

        if diabetic_idx == 0:
            glucose_state = np.random.choice(np.arange(5),
                                             p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = np.random.choice(np.arange(5),
                                             p=np.array([.01, .05, .15, .6, .19]))
        antibiotic_state = 0
        vaso_state = 0
        vent_state = 0

        state_categs = [hr_state, sysbp_state, percoxyg_state,
                        glucose_state, antibiotic_state, vaso_state, vent_state]

        return State(state_categs=state_categs, diabetic_idx=diabetic_idx)

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        """
        self.state.antibiotic_state = 1
        if self.state.hr_state == 2 and np.random.uniform(0, 1) < uniform_random(0.5, width=self.pkpd_noise*0.5):
            self.state.hr_state = 1
        if self.state.sysbp_state == 2 and np.random.uniform(0, 1) < uniform_random(0.5, width=self.pkpd_noise*0.5):
            self.state.sysbp_state = 1

    def transition_antibiotics_off(self):
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        """
        if self.state.antibiotic_state == 1:
            if self.state.hr_state == 1 and np.random.uniform(0, 1) < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.hr_state = 2
            if self.state.sysbp_state == 1 and np.random.uniform(0, 1) < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.sysbp_state = 2
            self.state.antibiotic_state = 0

    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        """
        self.state.vent_state = 1
        if self.state.percoxyg_state == 0 and np.random.uniform(0, 1) < uniform_random(0.7, width=self.pkpd_noise*0.5):
            self.state.percoxyg_state = 1

    def transition_vent_off(self):
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        """
        if self.state.vent_state == 1:
            if self.state.percoxyg_state == 1 and np.random.uniform(0, 1) < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.percoxyg_state = 0
            self.state.vent_state = 0

    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        """
        self.state.vaso_state = 1
        if self.state.diabetic_idx == 0:
            if np.random.uniform(0, 1) < uniform_random(0.7, width=self.pkpd_noise*0.5):
                if self.state.sysbp_state == 0:
                    self.state.sysbp_state = 1
                elif self.state.sysbp_state == 1:
                    self.state.sysbp_state = 2
        else:
            if self.state.sysbp_state == 1:
                if np.random.uniform(0, 1) < uniform_random(0.9, width=self.pkpd_noise*0.5):
                    self.state.sysbp_state = 2
            elif self.state.sysbp_state == 0:
                up_prob = np.random.uniform(0, 1)
                if up_prob < uniform_random(0.5, width=self.pkpd_noise*0.5):
                    self.state.sysbp_state = 1
                elif up_prob < uniform_random(0.9, width=self.pkpd_noise*0.5):
                    self.state.sysbp_state = 2
            if np.random.uniform(0, 1) < uniform_random(0.5, width=self.pkpd_noise*0.5):
                self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def transition_vaso_off(self):
        """
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        """
        if self.state.vaso_state == 1:
            if self.state.diabetic_idx == 0:
                if np.random.uniform(0, 1) < uniform_random(0.1, width=self.pkpd_noise*0.5):
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            else:
                if np.random.uniform(0, 1) < uniform_random(0.05, width=self.pkpd_noise*0.5):
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            self.state.vaso_state = 0

    def transition_fluctuate(self, hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
                             glucose_fluctuate):
        """
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        """
        if hr_fluctuate:
            hr_prob = np.random.uniform(0, 1)
            if hr_prob < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.hr_state = max(0, self.state.hr_state - 1)
            elif hr_prob < uniform_random(0.2, width=self.pkpd_noise*0.5):
                self.state.hr_state = min(2, self.state.hr_state + 1)
        if sysbp_fluctuate:
            sysbp_prob = np.random.uniform(0, 1)
            if sysbp_prob < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            elif sysbp_prob < uniform_random(0.2, width=self.pkpd_noise*0.5):
                self.state.sysbp_state = min(2, self.state.sysbp_state + 1)
        if percoxyg_fluctuate:
            percoxyg_prob = np.random.uniform(0, 1)
            if percoxyg_prob < uniform_random(0.1, width=self.pkpd_noise*0.5):
                self.state.percoxyg_state = max(0, self.state.percoxyg_state - 1)
            elif percoxyg_prob < uniform_random(0.2, width=self.pkpd_noise*0.5):
                self.state.percoxyg_state = min(1, self.state.percoxyg_state + 1)
        if glucose_fluctuate:
            glucose_prob = np.random.uniform(0, 1)
            if self.state.diabetic_idx == 0:
                if glucose_prob < uniform_random(0.1, width=self.pkpd_noise*0.5):
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < uniform_random(0.2, width=self.pkpd_noise*0.5):
                    self.state.glucose_state = min(1, self.state.glucose_state + 1)
            else:
                if glucose_prob < uniform_random(0.3, width=self.pkpd_noise*0.5):
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < uniform_random(0.6, width=self.pkpd_noise*0.5):
                    self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def calculateReward(self):
        num_abnormal = self.state.get_num_abnormal()
        if num_abnormal >= 3:
            return -1
        elif num_abnormal == 0 and not self.state.on_treatment():
            return 1
        return 0

    def transition(self, action):
        self.state = self.state.copy_state()

        if action.antibiotic == 1:
            self.transition_antibiotics_on()
            hr_fluctuate = False
            sysbp_fluctuate = False
        elif self.state.antibiotic_state == 1:
            self.transition_antibiotics_off()
            hr_fluctuate = False
            sysbp_fluctuate = False
        else:
            hr_fluctuate = True
            sysbp_fluctuate = True

        if action.ventilation == 1:
            self.transition_vent_on()
            percoxyg_fluctuate = False
        elif self.state.vent_state == 1:
            self.transition_vent_off()
            percoxyg_fluctuate = False
        else:
            percoxyg_fluctuate = True

        glucose_fluctuate = True

        if action.vasopressors == 1:
            self.transition_vaso_on()
            sysbp_fluctuate = False
            glucose_fluctuate = False
        elif self.state.vaso_state == 1:
            self.transition_vaso_off()
            sysbp_fluctuate = False

        self.transition_fluctuate(hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate,
                                  glucose_fluctuate)

        return self.calculateReward()

    def select_actions(self):
        assert self.policy_array is not None
        probs = self.policy_array[
            self.state.get_state_idx(self.policy_idx_type)
        ]
        aev_idx = np.random.choice(np.arange(Action.NUM_ACTIONS_TOTAL), p=probs)
        return Action(action_idx=aev_idx)


class SimSepsisEnv(gym.Env):
    """
    Sepsis environment for OpenAI Gym, most code is from https://github.com/clinicalml/gumbel-max-scm/
    state:
        full states are  diabetic_idx, heart rate, blood pressure, oxygen, glucose, antibiotic, vaso, vent, all of which
        are discrete. They respectively have 2, 3, 3, 2, 5, 2, 2, 2 states.
        The total number of states is 2*3*3*2*5*2*2*2 = 1440. In the full mode, all states are visible to the agent,
        while in the obs mode, diabetic idx is hidden from the agent.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, mode="proj_obs", max_t=40,
                 pkpd_noise=0.0,
                 obs_noise=0.2,
                 n_act=8,
                 p_diabetes=0.2,
                 missing_rate=0.0,
                 **kwargs):
        if n_act != 8:
            warnings.warn("n_act is not used. Action size is always 8.")
        if "state_noise" in kwargs:
            warnings.warn("state_noise is not used. The environment's initial state is always random.")

        self.action_space = gym.spaces.Discrete(8)
        self.mode = mode
        self.p_diabetes = p_diabetes
        self.max_t = max_t
        self.obs_noise = obs_noise
        self.pkpd_noise = pkpd_noise
        self.missing_rate = missing_rate

        assert mode in ["full", "obs", "proj_obs"]
        if mode == "full":
            self.observation_space = gym.spaces.Box(low=np.array([-1 - obs_noise] * 8).astype(np.float32),
                                                    high=np.array([1 + obs_noise] * 8).astype(np.float32))
        elif mode == "obs":
            self.observation_space = gym.spaces.Box(low=np.array([-1 - obs_noise] * 7).astype(np.float32),
                                                    high=np.array([1 + obs_noise] * 7).astype(np.float32))
        elif mode == "proj_obs":
            self.observation_space = gym.spaces.Box(low=np.array([-1 - obs_noise] * 6).astype(np.float32),
                                                    high=np.array([1 + obs_noise] * 6).astype(np.float32))
        else:
            raise ValueError("Mode {} not recognized".format(mode))

        self.terminate = False
        self.truncated = False
        self.info = {}
        self.steps = 0
        self.last_obs = None

        self.env_info = {'action_type': 'discrete', 'reward_range': (-1, 1),
                         "state_key": ["hr_state", "sysbp_state", "percoxyg_state", "glucose_state"
                                       "antibiotic_state", "vaso_state", "vent_state", "diabetic_idx"],
                         "obs_key": ["hr_state", "sysbp_state", "percoxyg_state", "glucose_state"],
                         "act_key": ["antibiotic", "ventilation", "vasopressors"]  # key for visu, not for forward
                         }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, ):
        if seed is not None:
            np.random.seed(seed)
        self.mdp = MDP(init_state_idx_type=self.mode, policy_idx_type=self.mode,
                       pkpd_noise=self.pkpd_noise, p_diabetes=self.p_diabetes)
        state = self.mdp.state
        self.steps = 0
        self.spec.reward_threshold = self.env_info['reward_range'][1]
        obs = self._state2obs(state, enable_missing=False).astype(np.float32)

        return obs,  {"full_state": self.mdp.state.get_state_idx(idx_type="full"),
                      "proj_state": self.mdp.state.get_state_idx(idx_type="proj_obs"),
                      "state": self._export_state(state),
                      "cur_act": [0, 0, 0],
                      "cur_act_idx": -1,
                      "step": self.steps,
                      "action": [0, 0, 0]}

    def seed(self, seed):
        super().reset(seed=seed)
        np.random.seed(seed)

    @staticmethod
    def discrete2conti(hr_state, sysbp_state, percoxyg_state, glucose_state, noise=0.2):
        """
        convert discrete states into continuous states for the four observational variables.
        A learnt word embedding is unnecessary.
        """
        try:
            # numpy array input
            if len(hr_state) != len(sysbp_state) != len(percoxyg_state) != len(glucose_state):
                raise ValueError("The length of the four states should be the same.")
            num = len(hr_state)
        except TypeError:
            pass  # single input

        # psudo continuous actions
        hr_state = (hr_state - 1)
        sysbp_state = (sysbp_state - 1)
        percoxyg_state = percoxyg_state
        glucose_state = (glucose_state - 2) / 2
        if noise:
            hr_state = uniform_random(hr_state, noise, absolute=True)
            sysbp_state = uniform_random(sysbp_state, noise, absolute=True)
            percoxyg_state = uniform_random(percoxyg_state, noise, absolute=True)
            glucose_state = uniform_random(glucose_state, noise, absolute=True)

        return hr_state, sysbp_state, percoxyg_state, glucose_state

    @staticmethod
    def state_idx_decompose(state_idx_, idx_type="full"):
        """
        returns integer index of state: significance order as in categorical array
        """

        if idx_type == 'obs':
            categ_num = [3, 3, 2, 5, 2, 2, 2]
            # order
            # hr_state,sysbp_state,percoxyg_state,glucose_state,antibiotic_state,vaso_state,vent_state

        elif idx_type == 'proj_obs':
            categ_num = [3, 3, 2, 2, 2, 2]
            # order
            # hr_state,sysbp_state,percoxyg_state,antibiotic_state,vaso_state,vent_state

        elif idx_type == 'full':
            categ_num = [2, 3, 3, 2, 5, 2, 2, 2]
            # order
            # diabetic_idx,hr_state,sysbp_state,percoxyg_state,glucose_state,antibiotic_state,vaso_state,vent_state

        else:
            raise ValueError('idx_type must be one of: obs, proj_obs, full')
        if state_idx_ < 0:
            return [np.nan for _ in range(len(categ_num))]
        states = []
        state_idx = int(state_idx_)
        for i in categ_num[::-1]:
            variable_idx = state_idx % i
            state_idx //= i
            states.append(variable_idx)
        states = states[::-1]

        # make sure the conversion is correct
        sum_idx = 0
        prev_base = 1
        for i in range(len(states)):
            idx = len(states) - 1 - i
            sum_idx += prev_base * states[idx]
            prev_base *= categ_num[idx]
        assert int(sum_idx) == int(state_idx_)
        return states

    @staticmethod
    def action_idx_decompose(action_idx):
        a = Action(action_idx=int(action_idx))
        return a.antibiotic, a.ventilation, a.vasopressors

    def get_all_state(self):
        return {"hr_state": self.mdp.state.hr_state, "sysbp_state": self.mdp.state.sysbp_state,
                "percoxyg_state": self.mdp.state.percoxyg_state, "glucose_state": self.mdp.state.glucose_state,
                "antibiotic_state": self.mdp.state.antibiotic_state, "vaso_state": self.mdp.state.vaso_state,
                "vent_state": self.mdp.state.vent_state, "diabetic_idx": self.mdp.state.diabetic_idx}

    def _export_state(self, state: State):
        hr_state, sysbp_state, percoxyg_state, glucose_state = self.discrete2conti(state.hr_state,
                                                                                   state.sysbp_state,
                                                                                   state.percoxyg_state,
                                                                                   state.glucose_state,
                                                                                   noise=0)
        antibiotic_state = state.antibiotic_state
        vaso_state = state.vaso_state
        vent_state = state.vent_state
        state = {"hr_state": hr_state, "sysbp_state": sysbp_state,
                 "percoxyg_state": percoxyg_state, "glucose_state": glucose_state,
                 "antibiotic_state": antibiotic_state, "vaso_state": vaso_state,
                 "vent_state": vent_state, "diabetic_idx": state.diabetic_idx}
        return state

    def _state2obs(self, state: State, enable_missing=False):
        """
        each vital state is categorical. Here we convert them into continuous values, rescale into [-1, 1] and insert
        a random noise.
        """
        diabetic_idx = state.diabetic_idx
        hr_state, sysbp_state, percoxyg_state, glucose_state = self.discrete2conti(state.hr_state,
                                                                                   state.sysbp_state,
                                                                                   state.percoxyg_state,
                                                                                   state.glucose_state,
                                                                                   noise=self.obs_noise)

        antibiotic_state = state.antibiotic_state
        vaso_state = state.vaso_state
        vent_state = state.vent_state

        hr_state, sysbp_state, percoxyg_state, glucose_state = float(hr_state), float(sysbp_state), \
            float(percoxyg_state), float(glucose_state)

        if self.mode == "full":
            obs = np.array([diabetic_idx, hr_state, sysbp_state, percoxyg_state,
                            glucose_state, antibiotic_state, vaso_state, vent_state])
        elif self.mode == "obs":
            obs = np.array([hr_state, sysbp_state, percoxyg_state,
                            glucose_state, antibiotic_state, vaso_state, vent_state])
        elif self.mode == "proj_obs":
            obs = np.array([hr_state, sysbp_state, percoxyg_state,
                            antibiotic_state, vaso_state, vent_state])
        else:
            raise ValueError("mode should be full, obs or proj_obs")

        if enable_missing and np.random.uniform(0, 1) < self.missing_rate:
            obs = self.last_obs
        else:
            self.last_obs = obs

        return obs

    def step(self, action: int):
        # action is a number between 0 and 7, which corresponds to the 8 actions

        self.steps += 1
        a = Action(action_idx=int(action))
        reward = self.mdp.transition(a)
        obs_next = self._state2obs(self.mdp.state, enable_missing=True)
        truncated = self.steps >= self.max_t
        terminate = reward != 0
        return (obs_next.astype(np.float32), reward, terminate, truncated,
                {"full_state": self.mdp.state.get_state_idx(idx_type="full"),
                 "proj_state": self.mdp.state.get_state_idx(idx_type="proj_obs"),
                 "state": self._export_state(self.mdp.state),
                 "cur_act": [a.antibiotic, a.ventilation, a.vasopressors],
                 "cur_act_idx": int(action),
                 "step": self.steps,
                 "instantaneous_reward": reward,
                 "action": [a.antibiotic, a.ventilation, a.vasopressors]})


def create_OberstSepsisEnv_discrete(max_t: int = 40, n_act: int = 8, **kwargs):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act, **kwargs)
    return env


def create_OberstSepsisEnv_discrete_setting1(max_t: int = 40, n_act: int = 8):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act,
                       pkpd_noise=0.0, obs_noise=0.0, missing_rate=0.0)
    return env


def create_OberstSepsisEnv_discrete_setting2(max_t: int = 40, n_act: int = 8):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act,
                       pkpd_noise=0.1, obs_noise=0.0, missing_rate=0.0)
    return env


def create_OberstSepsisEnv_discrete_setting3(max_t: int = 40, n_act: int = 8):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act,
                       pkpd_noise=0.1, obs_noise=0.2, missing_rate=0.0)
    return env


def create_OberstSepsisEnv_discrete_setting4(max_t: int = 40, n_act: int = 8):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act,
                       pkpd_noise=0.1, obs_noise=0.5, missing_rate=0.0)
    return env


def create_OberstSepsisEnv_discrete_setting5(max_t: int = 40, n_act: int = 8):
    env = SimSepsisEnv(mode="proj_obs", max_t=max_t, n_act=n_act,
                       pkpd_noise=0.1, obs_noise=0.5, missing_rate=0.5)
    return env


if __name__ == "__main__":
    env = gym.make("OberstSepsisEnv-discrete")
    for i in range(10000):
        obs, info = env.reset()
        done = False
        print(i)
        while not done:
            obs, reward, terminate, truncated, info = env.step(env.action_space.sample())
            done = terminate or truncated
