import optuna


def get_common_hparams(trial: optuna.trial.Trial, hparam_name):
    if hparam_name in ["lr",  "actor_lr", "critic_lr", "alpha_lr", "qf_lr"]:
        return trial.suggest_categorical(hparam_name, [1e-3, 1e-4, 1e-5, 1e-6])
    elif hparam_name == "batch_size":
        return trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    elif hparam_name == "stack_num":
        return trial.suggest_categorical("stack_num", [20, 50, 100])
    elif hparam_name == "batch_norm":
        return trial.suggest_categorical("batch_norm", [True, False])
    elif hparam_name == "dropout":
        return trial.suggest_categorical("dropout", [0, 0.25, 0.5])
    elif hparam_name == "target_update_freq":
        return trial.suggest_categorical("target_update_freq", [1, 1000, 5000])
    elif hparam_name == "update_per_step":
        return trial.suggest_categorical("update_per_step", [0.1, 0.5])
    elif hparam_name == "update_actor_freq":
        return trial.suggest_categorical("update_actor_freq", [1, 5, 10])
    elif hparam_name == "step_per_collect":
        return trial.suggest_categorical("step_per_collect", [50, 100])
    elif hparam_name == "n_step":
        trial.set_user_attr("n_step", 1)
        return 1
    elif hparam_name == "start_timesteps":
        trial.set_user_attr("start_timesteps", 0)
        return 0
    elif hparam_name == "gamma":
        trial.set_user_attr("gamma", 0.95)
        return 0.95
    elif hparam_name == "tau":
        trial.set_user_attr("tau", 0.001)
        return 0.001
    elif hparam_name == "exploration_noise":
        return trial.suggest_categorical("exploration_noise", [0.1, 0.2, 0.5])
    else:
        raise ValueError(f"Unknown hparam_name {hparam_name}")