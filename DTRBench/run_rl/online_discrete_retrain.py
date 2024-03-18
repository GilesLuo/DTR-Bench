import optuna
import argparse
import torch
from DTRBench.core.helper_fn import get_best_hparams, get_policy_class, get_hparam_class, get_obj_class, get_policy_type
from DTRBench.utils.misc import to_bool, early_stopping_callback
import DTRGym
import os

from pathlib import Path
from DTRBench.core.helper_fn import create_study_with_filter


def parse_args():
    parser = argparse.ArgumentParser()

    # training-aid hyperparameters
    parser.add_argument("--sampler", type=str, default="TPESampler", choices=["TPESampler", "BruteForceSampler"])
    parser.add_argument("--task", type=str, default="AhnChemoEnv")
    parser.add_argument("--setting", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="settings_db")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_actions", type=int, default=11)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--multi_obj", type=to_bool, default=False)
    parser.add_argument("--buffer_size", type=int, default=5e4)
    parser.add_argument("--linear", type=to_bool, default=False)
    parser.add_argument("--cat_num", type=int, default=1)
    parser.add_argument("--policy_name", type=str, default="C51-rnn",
                        choices=["DQN", "DDQN", "DQN-rnn",
                                 "DDQN-rnn", "DQN-dueling", "DDQN-dueling",
                                 "C51", "C51-rnn", "discrete-SAC", "discrete-SAC-rnn"])
    parser.add_argument("--scale_obs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()

    if "rnn" in args.policy_name:
        use_rnn = True
    else:
        use_rnn = False

    hparam_class = get_hparam_class(args.policy_name, offline=False)
    obj_class = get_obj_class(args.policy_name, offline=False)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    policy_type = get_policy_type(args.policy_name, offline=False)
    args.task += f"-{policy_type}-setting{args.setting}"
    if args.setting == 5:
        study_task_name = f"{args.task[:-1]}4"
    else:
        study_task_name = args.task
    study_name = f"{study_task_name}-{args.policy_name}"
    study_path = os.path.abspath(os.path.join(args.logdir, study_name)) + ".db"

    # init hparam space
    hparam_space = hparam_class(args.policy_name,
                                args.logdir,
                                args.seed,
                                args.training_num,  # number of training envs
                                args.test_num,  # number of test envs
                                args.epoch,
                                args.step_per_epoch,  # number of training steps per epoch
                                args.buffer_size,
                                use_rnn, args.num_actions,
                                cat_num=args.cat_num,
                                linear=args.linear
                                )
    # init obj
    obj = obj_class(args.task, hparam_space, device=args.device, multi_obj=args.multi_obj,
                    logger="tensorboard",
                    )

    user_attrs, hyperparameters, _ = get_best_hparams(study_path)
    user_attrs["logdir"] = args.logdir
    hyperparameters.update(user_attrs)
    result = obj.retrain_best_hparam(5000, **hyperparameters)
    print(result)
