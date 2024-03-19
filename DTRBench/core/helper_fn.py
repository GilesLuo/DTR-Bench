import optuna
from optuna import Study
import warnings
import os
import torch
import random
from typing import Union
from tianshou.policy import BasePolicy
from DTRBench.core.baseline_policy import RandomPolicy, MaxPolicy, MinPolicy

from tianshou.policy import DQNPolicy, C51Policy, DDPGPolicy, \
    TD3Policy, SACPolicy, REDQPolicy, DiscreteSACPolicy, DiscreteBCQPolicy, DiscreteCQLPolicy, BCQPolicy, CQLPolicy, \
    ImitationPolicy
from DTRBench.core.base_obj import RLObjective
from pathlib import Path
from DTRBench.core.offpolicyRLObj import DQNObjective, C51Objective, DDPGObjective, SACObjective, TD3Objective, \
    REDQObjective, DiscreteSACObjective
from DTRBench.core.base_obj import OffPolicyRLHyperParameterSpace
from DTRBench.core.offpolicyRLHparams import DQNHyperParams, C51HyperParams, DDPGHyperParams, SACHyperParams, \
    TD3HyperParams, REDQHyperParams
import os
import shutil
import optuna


def create_study_with_filter(study_name, study_path, direction, sampler, load_if_exists=True, pruner=None):
    # Step 1: Check if tmp.db exists
    tmp_db_path = f"tmp_{study_name}.db"
    storage_url = study_url = f"sqlite:///{study_path}"
    if os.path.exists(tmp_db_path):
        raise FileExistsError(f"{tmp_db_path} already exists. Please remove it before proceeding.")

    original_study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction,
                                         sampler=sampler, pruner=pruner, load_if_exists=load_if_exists)

    # Step 2: Filter out failed trials and create a tmp study in tmp_{study_name}.db
    successful_trials = [t for t in original_study.trials if t.state != optuna.trial.TrialState.FAIL]

    if len(successful_trials) == len(original_study.trials):
        print(f"Study '{study_name}' has no failed trials, no need to overwrite.")
        return original_study

    tmp_storage_url = f"sqlite:///{tmp_db_path}"
    tmp_study = optuna.create_study(study_name=study_name, storage=tmp_storage_url, direction=direction,
                                    sampler=sampler, pruner=pruner)

    for trial in successful_trials:
        tmp_study.add_trial(trial)

    os.remove(study_path)
    shutil.move(tmp_db_path, study_path)
    print(f"Study '{study_name}' has been overwritten with successful trials only.")

    # Step 3: Create a new study with the original storage
    new_study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction,
                                    sampler=sampler, pruner=pruner, load_if_exists=load_if_exists)
    return new_study


def get_best_hparams(study=Union[Study, str], first_n=None):
    if isinstance(study, str):
        database_path = os.path.abspath(study)
        try:
            study = optuna.load_study(study_name=Path(os.path.basename(study)).stem,
                                      storage=f"sqlite:///{database_path}")
        except KeyError:
            try:
                study = optuna.load_study(study_name=None,
                                          storage=f"sqlite:///{database_path}")
            except Exception:
                raise ValueError(f"Could not load study from {database_path}")
    elif isinstance(study, Study):
        pass
    else:
        raise ValueError(f"study must be either a string or an optuna.Study object, got {type(study)}")
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]
    if first_n is None:
        first_n = len(trials_df)
    try:
        df = trials_df.iloc[:first_n, :]
    except KeyError:
        warnings.warn(f"No enough completed trials found for first_n={first_n}, using all {len(trials_df)} trials")
        df = trials_df
    best_trial_number = df.loc[df["value"].idxmax(), "number"]
    best_trial = study.trials[best_trial_number]
    hyperparameters = best_trial.params
    user_attrs = best_trial.user_attrs
    return user_attrs, hyperparameters, best_trial.value


def filter_hparams(hparams, hparam_class: OffPolicyRLHyperParameterSpace):
    def f(dict1, keys):
        return {k: dict1[k] for k in keys if k in dict1.keys()}

    meta_keys = hparam_class._meta_hparams
    general_keys = hparam_class._general_hparams
    policy_keys = hparam_class._policy_hparams
    return f(hparams, meta_keys), f(hparams, general_keys), f(hparams, policy_keys)


def fetch_policy_instance(algo_name: str, offline: bool, env_name: str,
                          study_path: str, n_trials: int, device: str):
    # get best hyperparameters
    user_attrs, hyperparameters, _ = get_best_hparams(study_path, first_n=n_trials)
    hyperparameters.update(user_attrs)

    # get policy and env
    hparam_class = get_hparam_class(algo_name, offline=offline)
    obj_class: RLObjective = get_obj_class(algo_name, offline)

    meta_hp, general_hp, policy_hp = filter_hparams(hyperparameters, hparam_class)

    hparam_space = hparam_class(**meta_hp)
    obj = obj_class(env_name, hparam_space, device)
    policy = obj.define_policy(**hyperparameters)
    return policy, meta_hp, general_hp, policy_hp


def policy_load(policy, ckpt_path: str, device: str, is_train: bool = False):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        ckpt = ckpt if ckpt_path.endswith("policy.pth") else ckpt["model"]  # policy.pth and ckpt.pth has different keys
        policy.load_state_dict(ckpt)
    if is_train:
        policy.train()
    else:
        policy.eval()
    return policy


offpolicyLOOKUP = {
    "dqn": {"hparam": DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
    "ddqn": {"hparam": DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
    "c51": {"hparam": C51HyperParams, "policy": C51Policy, "obj": C51Objective, "type": "discrete"},
    "c51-rnn": {"hparam": C51HyperParams, "policy": C51Policy, "obj": C51Objective, "type": "discrete"},
    "discrete-sac": {"hparam": SACHyperParams, "policy": DiscreteSACPolicy, "obj": DiscreteSACObjective,
                     "type": "discrete"},
    "discrete-sac-rnn": {"hparam": SACHyperParams, "policy": DiscreteSACPolicy, "obj": DiscreteSACObjective,
                         "type": "discrete"},
    "ddpg": {"hparam": DDPGHyperParams, "policy": DDPGPolicy, "obj": DDPGObjective, "type": "continuous"},
    "sac": {"hparam": SACHyperParams, "policy": SACPolicy, "obj": SACObjective, "type": "continuous"},
    "td3": {"hparam": TD3HyperParams, "policy": TD3Policy, "obj": TD3Objective, "type": "continuous"},
    "redq": {"hparam": REDQHyperParams, "policy": REDQPolicy, "obj": REDQObjective, "type": "continuous"},
}

BASELINE_LOOKUP = {"random": {"policy": RandomPolicy},
                   "max": {"policy": MaxPolicy},
                   "min": {"policy": MinPolicy}
                   }


def get_policy_class(algo_name) -> BasePolicy:
    algo_name = algo_name.lower()
    if "dqn" in algo_name:
        algo_name = "dqn"
    elif "ddqn" in algo_name:
        algo_name = "ddqn"
    elif "discrete-imitation" in algo_name:
        algo_name = "discrete-imitation"
    return offpolicyLOOKUP[algo_name]["policy"]


def get_hparam_class(algo_name: str, offline) -> OffPolicyRLHyperParameterSpace.__class__:
    algo_name = algo_name.lower()
    if "dqn" in algo_name:
        algo_name = "dqn"
    elif "ddqn" in algo_name:
        algo_name = "ddqn"
    elif "discrete-imitation" in algo_name:
        algo_name = "discrete-imitation"

    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["hparam"]


def get_obj_class(algo_name: str, offline) -> RLObjective.__class__:
    algo_name = algo_name.lower()
    if "dqn" in algo_name:
        algo_name = "dqn"
    elif "ddqn" in algo_name:
        algo_name = "ddqn"
    elif "discrete-imitation" in algo_name:
        algo_name = "discrete-imitation"
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["obj"]


def get_policy_type(algo_name: str, offline: bool) -> str:
    algo_name = algo_name.lower()
    if "dqn" in algo_name:
        algo_name = "dqn"
    elif "ddqn" in algo_name:
        algo_name = "ddqn"
    elif "discrete-imitation" in algo_name:
        algo_name = "discrete-imitation"
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["type"]


def get_baseline_policy_class(algo_name: str) -> BasePolicy:
    return BASELINE_LOOKUP[algo_name]["policy"]


def retrieve_filepath(target_folder, extension):
    """
    :return: all files path relative to target_folder (including in the sub-folder) with the given extension in the target folder
    """
    file_list = [str(file.relative_to(target_folder)) for file in Path(target_folder).rglob(f'*{extension}')]

    return file_list
