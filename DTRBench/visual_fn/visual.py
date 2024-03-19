import gymnasium as gym
import os
import torch
import numpy as np
from pathlib import Path

from DTRBench.visual_fn.env_vis_fn import visualise_env_policy_fn
from DTRBench.core.baseline_policy import AllMinPolicy
from DTRBench.utils.get_policy import get_DQN, get_DDQN, get_DDQN_dueling, get_C51,\
    get_discrete_SAC, get_DQN_rnn, get_DDQN_rnn, get_C51_rnn, get_discrete_SAC_rnn,\
    get_DDPG, get_TD3, get_SAC


policy_load_fns = {"DQN": get_DQN,
                   "DDQN": get_DDQN,
                   "DDQN-dueling": get_DDQN_dueling,
                   "C51": get_C51,
                   "discrete-SAC": get_discrete_SAC,
                   "DQN-rnn": get_DQN_rnn,
                   "DDQN-rnn": get_DDQN_rnn,
                   "C51-rnn": get_C51_rnn,
                   "discrete-SAC-rnn": get_discrete_SAC_rnn,
                   "DDPG": get_DDPG,
                   "TD3": get_TD3,
                   "SAC": get_SAC,
                   "zero-drug": None}


def visual_main_fn(seed, env_name, policy_name, n_act,
                   plot_save_dir, plot_format="pdf"):
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    Path(plot_save_dir).mkdir(parents=True, exist_ok=True)

    # init env
    env = gym.make(env_name, n_act=n_act)

    # init policy
    obs_shape = env.observation_space.shape

    if policy_name == "zero-drug":
        act_min = env.action_space.low if isinstance(env.action_space, gym.spaces.Box) else 0
        policy = AllMinPolicy(act_min=act_min)
    else:
        policy_weight_path = os.path.join("settings_db", f"{env_name}/{policy_name}-best/{seed}", "policy.pth")
        policy_load_fn = policy_load_fns[policy_name]
        if 'discrete' in env_name:
            policy = policy_load_fn(obs_shape, n_act, weights_path=policy_weight_path)
        else:
            policy = policy_load_fn(obs_shape, action_space=env.action_space, weights_path=policy_weight_path)

    plot_save_path = os.path.join(plot_save_dir,
                                  f"{env_name}_{policy_name}.{plot_format}")

    visualise_env_policy_fn(env, policy, num_episodes=1,
                              plot_save_path=plot_save_path,
                              use_log_scale=False,
                              drop_last_step=False)


if __name__ == "__main__":
    seed = 21243
    plot_save_dir = "plots"
    env_name = "GhaffariCancerEnv-discrete-setting1"
    discrete_policy_names = ["DQN", "DDQN", "DDQN-dueling", "C51", "discrete-SAC",
                             "DQN-rnn", "DDQN-rnn", "C51-rnn", "discrete-SAC-rnn",
                             "zero-drug"]
    continuous_policy_names = ["DDPG", "TD3", "SAC"]
    n_act = 25  # only for discrete action space

    if 'discrete' in env_name:
        policy_names = discrete_policy_names
    elif 'continuous' in env_name:
        policy_names = continuous_policy_names

    for policy_name in policy_names:
        visual_main_fn(seed=seed,
                       env_name=env_name,
                       policy_name=policy_name,
                       n_act=n_act,
                       plot_save_dir=plot_save_dir)
