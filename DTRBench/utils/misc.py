import optuna
import torch
import random
import numpy as np



def to_bool(value):
    valid = {'true': True, 't': True, '1': True,
             'false': False, 'f': False, '0': False,
             }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def set_global_seed(seed):
    # Set seed for Python's built-in random
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for all GPU devices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set seed for torch DataLoader
    def _worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn

def early_stopping_callback(study, trial):
    trials_df = study.trials_dataframe()
    completed_trials = len(trials_df[trials_df["state"] == "COMPLETE"])

    # Number of trials to check for repetitions
    n_trials_to_check = 3

    # If fewer trials than n_trials_to_check have been completed, don't stop
    if completed_trials < n_trials_to_check:
        return

    # Get the last n_trials_to_check trials
    recent_trials = study.trials[-n_trials_to_check:]

    # Check if objective values are the same for all of them
    is_repeated = all(trial.value == recent_trials[0].value for trial in recent_trials)

    if is_repeated:
        print(f"Stopping optimization because the objective value repeated for the last {n_trials_to_check} trials.")
        raise optuna.TrialPruned()
