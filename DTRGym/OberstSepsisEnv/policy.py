from tianshou.policy import BasePolicy
import os
from .state import State
import numpy as np
import warnings
from tianshou.data import Batch
from typing import Union, Optional, Tuple, Any, Dict, Callable, List

warnings.simplefilter(action='ignore', category=FutureWarning)


# This is a QA function to ensure that the RL policy is only taking actions that have been observed
def check_rl_policy(rl_policy, obs_samps, proj_lookup, r_mat, NSTEPS, NSIMSAMPS):
    passes = True
    # Check the observed actions for each state
    obs_pol = np.zeros_like(rl_policy)
    for eps_idx in range(NSIMSAMPS):
        for time_idx in range(NSTEPS):
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])
            # Need to get projected state
            if this_obs_action == -1:
                continue
            this_obs_state = proj_lookup[int(obs_samps[eps_idx, time_idx, 2])]
            obs_pol[this_obs_state, this_obs_action] += 1

    # Check if each RL action conforms to an observed action
    for eps_idx in range(NSIMSAMPS):
        for time_idx in range(NSTEPS):
            this_full_state_unobserved = int(obs_samps[eps_idx, time_idx, 1])
            this_obs_state = proj_lookup[this_full_state_unobserved]
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])

            if this_obs_action == -1:
                continue
            # This is key: In some of these trajectories, you die or get discharge.
            # In this case, no action is taken because the sequence has terminated, so there's nothing to compare the RL action to
            true_death_states = r_mat[0, 0, 0, :] == -1
            true_disch_states = r_mat[0, 0, 0, :] == 1
            if np.logical_or(true_death_states, true_disch_states)[this_full_state_unobserved]:
                continue

            this_rl_action = rl_policy[proj_lookup[this_obs_state]].argmax()
            if obs_pol[this_obs_state, this_rl_action] == 0:
                print("Eps: {} \t RL Action {} in State {} never observed".format(
                    int(time_idx / NSTEPS), this_rl_action, this_obs_state))
                passes = False
    return passes


def projection_func_wrapper(proj_lookup):
    def projection_func(obs_state_idx):
        if obs_state_idx == -1:
            return -1
        else:
            return proj_lookup[obs_state_idx]

    return projection_func


class Optimal(BasePolicy):
    def __init__(self, fullPol_file: str):
        """
        :param fullPol_file: a npz file containing the full policy matrix with the shape of [1440, 8]
        """
        super().__init__()
        self.fullPol = np.load(fullPol_file)["fullPol"]

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        # 1. get the full state
        state_dict = batch.info["full_state"]
        state_indices = []
        for b in len(batch):
            state_idx = State(state_categs=[state_dict["hr_state"][b],
                                            state_dict["sysbp_state"][b],
                                            state_dict["percoxyg_state"][b],
                                            state_dict["glucose_state"][b],
                                            state_dict["antibiotic_state"][b],
                                            state_dict["vaso_state"][b],
                                            state_dict["vent_state"]][b],
                              diabetic_idx=state_dict["diabetic_idx"][b], )

            state_indices.append(state_idx.get_state_idx(idx_type="full_state"))
        # 2. get the action from the full policy
        # 3. return the action

        return Batch(act=self.fullPol[state_indices].argmax(), state=state)
