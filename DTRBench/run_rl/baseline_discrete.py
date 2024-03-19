import sys
import numpy as np
import torch
from DTRBench.core.helper_fn import get_baseline_policy_class
from DTRGym.base import make_env
from tianshou.data import Collector

def test_baseline_online(env_name, policy_name, test_seed, test_num, num_actions):
    if policy_name == "all":
        from DTRBench.core.helper_fn import BASELINE_LOOKUP

        policy_names = list(BASELINE_LOOKUP.keys())
        result = {}
        for p in policy_names:
            result[f"{p}-online"] = test_baseline_online(env_name, p, test_seed, test_num, num_actions)
    else:
        env, _, test_envs = make_env(env_name, int(test_seed), 1, 1, num_actions)
        policy_cls = get_baseline_policy_class(policy_name)
        policy = policy_cls(env.action_space)
        collector = Collector(policy, test_envs)
        result = collector.collect(n_episode=test_num, render=False)
        print(f"{args.policy_name} policy result: mean_reward:{result['rew']}, "
              f"reward_std:{result['rews'].std()}, mean_len:{result['len']}, len_std{result['lens'].std()}")
        result = {f"{policy_name}-online": result}
    return result


def test_baseline_offline(env_name, policy_name, train_buffer_name, test_buffer_keyword, test_seed, test_num,
                          num_actions, OPE_names):
    raise NotImplementedError("This function is not implemented yet")

# Function to format the mean and std values to two decimal places
def format_mean_std(mean, std):
    mean_formatted = "{:.2f}".format(mean)
    std_formatted = "{:.2f}".format(std)
    return f"{mean_formatted} ± {std_formatted}"

def result_to_latex(mean_df, std_df):
    # Function to format the mean and std values to two decimal places
    def format_mean_std(mean, std):
        mean_formatted = "{:.2f}".format(mean)
        std_formatted = "{:.2f}".format(std)
        return f"{mean_formatted} ± {std_formatted}"

    # Combining the mean and standard deviation dataframes
    combined_df = mean_df.copy()
    for col in combined_df.columns:
        combined_df[col] = combined_df.apply(
            lambda row: format_mean_std(float(row[col]), float(std_df[col][row.name])), axis=1)

    # Transposing the updated dataframe
    transposed_df_updated = combined_df.T

    # Converting the transposed dataframe to a LaTeX table
    latex_table_transposed_updated = transposed_df_updated.to_latex(header=True)
    return latex_table_transposed_updated

if __name__ == "__main__":
    import argparse

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # training-aid hyperparameters

    parser.add_argument("--env", type=str, default="MIMIC3SepsisEnv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=11)
    parser.add_argument("--policy_name", type=str, default="all", choices=["all", "random"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_buffer", type=str, default="train")
    parser.add_argument("--test_buffer_keyword", type=str, default="test", help="keyword to find all test buffer")
    parser.add_argument("--online", type=bool, default=False)
    parser.add_argument("--OPE_methods", nargs='+', choices=['IS', 'FQE', 'WIS', 'WIS_bootstrap',
                                                             'WIS_truncated', 'WIS_bootstrap_truncated',
                                                             'PatientWiseF1', "SampleWiseF1"],
                        default=['WIS', 'WIS_bootstrap', 'PatientWiseF1', "SampleWiseF1"],
                        help="Select one or more options from the list")

    args = parser.parse_known_args()[0]

    args.env += "-discrete"
    np.random.seed(args.seed)
    _ = np.random.randint(0, 100000, 5)  # not used, just for consistency with model testing
    test_seed = np.random.randint(0, 100000, 1)[0]
    if args.online:
        result = test_baseline_online(args.env, args.policy_name, test_seed, args.test_num, args.num_actions)
    else:
        mean, std = test_baseline_offline(args.env, args.policy_name, args.train_buffer, args.test_buffer_keyword, test_seed,
                              args.test_num, args.num_actions, args.OPE_methods)

    print()
