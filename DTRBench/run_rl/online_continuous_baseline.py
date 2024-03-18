import optuna
import numpy as np
import torch
from DTRBench.core.baseline_policy import RandomPolicy, MinPolicy, MaxPolicy, eval_baseline_policy

import os
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="AhnChemoEnv")
    parser.add_argument("--setting", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="settings_db")

    parser.add_argument("--num_seed", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=5000)
    parser.add_argument("--policy_name", type=str, default="max", choices=["min", 'max', 'random'])

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()

    policy_name = args.policy_name
    env_name = f"{args.task}-continuous-setting{args.setting}"
    result = eval_baseline_policy(policy_name=policy_name,
                                  env_name=env_name,
                                  log_dir=args.logdir,
                                  test_num=args.test_num,
                                  num_seed=args.num_seed)
    print("final evaluation:")
    print(result)
    