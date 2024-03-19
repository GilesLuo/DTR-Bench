import gymnasium as gym
import os
import torch
import numpy as np
from pathlib import Path

import DTRGym
from DTRBench.core.helper_fn import get_best_hparams,  get_hparam_class, get_obj_class
from DTRBench.visual_fn.env_vis_fn import visualise_env_policy_fn
from DTRBench.utils.get_policy import get_DQN


def demo_train_policy(log_dir="demo"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    policy_name = "DQN"
    task = "AhnChemoEnv-discrete-setting1"

    hparam_class = get_hparam_class(policy_name, offline=False)
    obj_class = get_obj_class(policy_name, offline=False)

    study_name = f"{task}-{policy_name}"
    study_path = os.path.abspath(os.path.join("demo", study_name)) + ".db"

    hparam_space = hparam_class(policy_name,
                                log_dir,
                                0,
                                1,  # number of training envs
                                1,  # number of test envs
                                1,  # epoch
                                100,  # number of training steps per epoch
                                1000,  # buffer size
                                cat_num=1,
                                num_actions=11,
                                use_rnn=False,
                                linear=False)

    obj = obj_class(task, hparam_space,
                    logger="tensorboard", device="cpu", multi_obj=False)

    # retrain the best model
    user_attrs, hyperparameters, _ = get_best_hparams(study_path)
    user_attrs["logdir"] = log_dir
    hyperparameters.update(user_attrs)
    hyperparameters["epoch"] = 1  # retrain for 1 epoch only for demo
    result = obj.retrain_best_hparam(100, num_seed=1, **hyperparameters)
    print("final evaluation:")
    print(result)

def demo_visual(log_dir="demo"):
    policy_name = "DQN"
    task = "AhnChemoEnv-discrete-setting1"

    # initialise the environment
    n_act = 11
    env = gym.make(task, n_act=n_act)

    # initialise the policy
    obs_shape = env.observation_space.shape
    weights_path = os.path.join(log_dir, task, f"{policy_name}-best", "68268", "policy.pth")
    policy = get_DQN(obs_shape=obs_shape,
                     action_shape=n_act,
                     weights_path=weights_path)

    # visualise the policy
    plot_save_path = os.path.join(log_dir, f"demo_{task}-{policy_name}.png")
    visualise_env_policy_fn(env, policy, plot_save_path=plot_save_path, num_episodes=1)


if __name__ == "__main__":
    demo_train_policy()
    demo_visual()
