import optuna
from DTRBench.core.base_hparams import get_common_hparams


class OffPolicyRLHyperParameterSpace:
    _meta_hparams = [
        "algo_name",  # name of the algorithm
        "logdir",  # directory to save logs
        "seed",
        "training_num",  # number of training envs
        "test_num",  # number of test envs
        "epoch",
        "step_per_epoch",  # number of training steps per epoch
        "buffer_size",  # size of replay buffer
        "use_rnn",
        "num_actions",  # number of actions, only used for discrete action space
        "cat_num",  # number of frames to concatenate, cannot be used with stack_num or rnn
        "linear",  # whether to use linear approximation as network
    ]

    _general_hparams = (
        # general parameters
        "batch_size",
        "step_per_collect",  # number of steps per collect. refer to tianshou's doc
        "update_per_step",
        "stack_num",  # number of frames to stack
        "gamma",
    )
    _policy_hparams = ()
    _supported_algos = ()

    def __init__(self,
                 algo_name,  # name of the algorithm
                 logdir,  # directory to save logs
                 seed,
                 training_num,  # number of training envs
                 test_num,  # number of test envs
                 epoch,
                 step_per_epoch,  # number of training steps per epoch
                 buffer_size,  # size of replay buffer
                 use_rnn,
                 num_actions=None,  # number of actions, only used for discrete action space
                 cat_num=1,
                 linear=False
                 ):
        if algo_name.lower() not in [i.lower() for i in self.__class__._supported_algos]:
            raise NotImplementedError(f"algo_name {algo_name} not supported, support {self.__class__._supported_algos}")
        self.algo_name = algo_name
        self.logdir = logdir
        self.seed = seed
        self.training_num = training_num
        self.test_num = test_num
        self.epoch = epoch
        self.step_per_epoch = step_per_epoch
        self.buffer_size = buffer_size
        self.use_rnn = use_rnn
        self.num_actions = num_actions
        self.cat_num = cat_num
        self.linear = linear

    def check_illegal(self):
        """
        all hyperparameters should be defined in _meta_hparams, _general_hparams and _policy_hparams. If not, raise error
        and list the undefined hyperparameters
        :return: list of undefined hyperparameters
        """
        all_hparams = list(self._meta_hparams) + list(self._general_hparams) + list(self._policy_hparams)
        undefined_hparams = [h for h in all_hparams if not hasattr(self, h)]
        keys = ["_meta_hparams", "_general_hparams", "_policy_hparams", "_supported_algos"]
        unknown_hparams = [h for h in self.__dict__() if h not in all_hparams and h not in keys]
        if len(undefined_hparams) > 0:
            printout1 = f"undefined hyperparameters: {undefined_hparams}"
        else:
            printout1 = ""
        if len(unknown_hparams) > 0:
            printout2 = f"unknown hyperparameters: {unknown_hparams}"
        else:
            printout2 = ""
        if len(printout1) > 0 or len(printout2) > 0:
            raise ValueError(f"{printout1}\n{printout2}")

    def get_meta_params(self):
        return {k: getattr(self, k) for k in self._meta_hparams}

    def get_general_params(self):
        return {k: getattr(self, k) for k in self._general_hparams}

    def get_policy_params(self):
        return {k: getattr(self, k) for k in self._policy_hparams}

    def get_all_params(self):
        result = {}
        dict_args = [self.get_meta_params(), self.get_general_params(), self.get_policy_params()]
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def define_general_hparams(self, trial: optuna.trial.Trial):
        # define general hparams

        self.batch_size = get_common_hparams(trial, "batch_size")
        self.step_per_collect = get_common_hparams(trial, "step_per_collect")
        self.update_per_step = get_common_hparams(trial, "update_per_step")
        self.gamma = get_common_hparams(trial, "gamma")
        if self.use_rnn:
            self.stack_num = get_common_hparams(trial, "stack_num")
        else:
            self.stack_num = 1
            trial.set_user_attr("stack_num", 1)
        return {
            "batch_size": self.batch_size,
            "step_per_collect": self.step_per_collect,
            "update_per_step": self.update_per_step,
            "gamma": self.gamma,
            "stack_num": self.stack_num,
        }

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        raise NotImplementedError

    def __call__(self, trial: optuna.trial.Trial):
        # define meta hparams
        for p in self._meta_hparams:
            trial.set_user_attr(p, getattr(self, p))

        # define general hparams
        meta_hparams = self.get_meta_params()
        general_hparams = self.define_general_hparams(trial)
        policy_hparams = self.define_policy_hparams(trial)
        self.check_illegal()
        result = {}
        dict_args = [meta_hparams, general_hparams, policy_hparams]
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def keys(self):
        return self.__dict__()

    def __dict__(self):
        return {k for k in dir(self) if not k.startswith('__') and not callable(getattr(self, k))}

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for key in dir(self):
            if not key.startswith('__') and not callable(getattr(self, key)):
                yield key, getattr(self, key)

    def __str__(self):
        # This will combine the dict representation with the class's own attributes
        class_attrs = {k: getattr(self, k) for k in dir(self) if
                       not k.startswith('__') and not callable(getattr(self, k))}
        all_attrs = {**self, **class_attrs}
        return str(all_attrs)

class DQNHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("dqn", "ddqn",
                        "dqn-rnn", "ddqn-rnn",
                        "dqn-dueling", "ddqn-dueling",
                        )
    _policy_hparams = (
        "lr",  # learning rate
        "eps_test",
        "eps_train",
        "eps_train_final",
        "n_step",
        "target_update_freq",
        "is_double",
        "use_dueling",)

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        # dqn hp
        self.n_step = get_common_hparams(trial, "n_step")
        self.target_update_freq = get_common_hparams(trial, "target_update_freq")
        self.lr = get_common_hparams(trial, "lr")

        eps_test = 0.005
        eps_train = 1
        eps_train_final = 0.005

        self.eps_test = eps_test
        self.eps_train = eps_train
        self.eps_train_final = eps_train_final

        self.is_double = True if "ddqn" in self.algo_name.lower() else False
        self.use_dueling = True if "dueling" in self.algo_name.lower() else False

        trial.set_user_attr("is_double", self.is_double)
        trial.set_user_attr("use_dueling", self.use_dueling)
        trial.set_user_attr("eps_test", eps_test)
        trial.set_user_attr("eps_train", eps_train)
        trial.set_user_attr("eps_train_final", eps_train_final)

        trial.set_user_attr("icm_lr_scale", 0.)
        trial.set_user_attr("icm_reward_scale", 0.)
        trial.set_user_attr("icm_forward_loss_weight", 0.)
        return {
            "lr": self.lr,
            "n_step": self.n_step,
            "target_update_freq": self.target_update_freq,
            "is_double": self.is_double,
            "use_dueling": self.use_dueling,
            "eps_test": self.eps_test,
            "eps_train": self.eps_train,
            "eps_train_final": self.eps_train_final,
            "icm_lr_scale": 0.,
            "icm_reward_scale": 0.,
            "icm_forward_loss_weight": 0.
        }


class DQN_ICMHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("dqn", "ddqn")
    _policy_hparams = ("dropout",
                       "n_step",
                       "target_update_freq",
                       "icm_lr_scale",
                       "icm_reward_scale",
                       "icm_forward_loss_weight")

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        self.dropout = get_common_hparams(trial, "dropout")

        # dqn hp
        self.n_step = get_common_hparams(trial, "n_step")
        self.target_update_freq = get_common_hparams(trial, "target_update_freq")

        self.icm_lr_scale = trial.suggest_categorical("icm_lr_scale",
                                                      [0.05, 0.1, 0.5, 1])  # help="use intrinsic curiosity module with this lr scale"
        self.icm_reward_scale = trial.suggest_categorical("icm_lr_scale",
                                                          [0.05, 0.1, 0.5, 1])  # help="scaling factor for intrinsic curiosity reward"
        self.icm_forward_loss_weight = trial.suggest_categorical("icm_lr_scale",
                                                                 [0.05, 0.1, 0.5, 1])  # help="weight for the forward model loss in ICM",
        trial.set_user_attr("n_step", self.n_step)
        return {"dropout": self.dropout,
                "n_step": self.n_step,
                "target_update_freq": self.target_update_freq,
                "icm_lr_scale": self.icm_lr_scale,
                "icm_reward_scale": self.icm_reward_scale,
                "icm_forward_loss_weight": self.icm_forward_loss_weight
                }


class C51HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("c51", "c51-rnn")
    _policy_hparams = ("lr",
                       "num_atoms",
                       "v_min",
                       "v_max",
                       "estimation_step",
                       "target_update_freq")

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        # c51 hp

        self.lr = get_common_hparams(trial, "lr")
        self.v_min = trial.suggest_categorical("v_min", [-20, -10, -5])
        self.v_max = trial.suggest_categorical("v_max", [5, 10, 20])
        self.target_update_freq = get_common_hparams(trial, "target_update_freq")

        self.num_atoms = 51
        self.eps_test = 0.005
        self.eps_train = 1
        self.eps_train_final = 0.005
        self.estimation_step = 1

        trial.set_user_attr("num_atoms",  self.num_atoms)
        trial.set_user_attr("eps_test", self.eps_test)
        trial.set_user_attr("eps_train", self.eps_train)
        trial.set_user_attr("eps_train_final", self.eps_train_final)
        trial.set_user_attr("estimation_step", self.estimation_step)
        return {"lr": self.lr,
                "num_atoms": self.num_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "estimation_step": self.estimation_step,
                "target_update_freq": self.target_update_freq,
                "eps_test": self.eps_test,
                "eps_train": self.eps_train,
                "eps_train_final": self.eps_train_final
                }


class DDPGHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("ddpg",)
    _policy_hparams = (
        "actor_lr",
        "critic_lr",
        "n_step",
        "exploration_noise",
        "tau",
        "start_timesteps"
    )

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        # dqn hp
        self.actor_lr = get_common_hparams(trial, "actor_lr")
        self.critic_lr = get_common_hparams(trial, "critic_lr")
        self.n_step = get_common_hparams(trial, "n_step")
        self.start_timesteps = get_common_hparams(trial, "start_timesteps")
        self.tau = get_common_hparams(trial, "tau")
        self.exploration_noise = get_common_hparams(trial, "exploration_noise")
        return {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "exploration_noise": self.exploration_noise,
            "tau": self.tau,
            "n_step": self.n_step,
            "start_timesteps": self.start_timesteps,
        }


class REDQHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("redq",)
    _policy_hparams = (
        "actor_lr",
        "critic_lr",
        "ensemble_size",
        "subset_size",
        "tau",
        "alpha",
        "actor_delay",
        "exploration_noise"
    )

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        # redq hp
        self.actor_lr = get_common_hparams(trial, "actor_lr")
        self.critic_lr = get_common_hparams(trial, "critic_lr")
        self.subset_size = 5
        trial.set_user_attr("subset_size", self.subset_size)
        self.exploration_noise = get_common_hparams(trial, "exploration_noise")
        self.alpha = trial.suggest_categorical("alpha", [0.05, 0.2, 0.4])
        self.tau = get_common_hparams(trial, "tau")
        self.ensemble_size = 10
        self.actor_delay = 20
        trial.set_user_attr("actor_delay", self.actor_delay)
        trial.set_user_attr("ensemble_size", self.ensemble_size)
        self.start_timesteps = get_common_hparams(trial, "start_timesteps")
        return {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "ensemble_size": self.ensemble_size,
            "subset_size": self.subset_size,
            "tau": self.tau,
            "start_timesteps": self.start_timesteps,
            "alpha": self.alpha,
            "actor_delay": self.actor_delay,
            "exploration_noise": self.exploration_noise}


class SACHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("sac", "discrete-sac", "discrete-sac-rnn")
    _policy_hparams = (
        "actor_lr",
        "critic_lr",
        "alpha",
        "n_step",
        "tau",
        "start_timesteps",

    )

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        # dqn hp
        self.actor_lr = get_common_hparams(trial, "actor_lr")
        self.critic_lr = get_common_hparams(trial, "critic_lr")
        self.alpha = trial.suggest_categorical("alpha", [0.05, 0.1, 0.2])
        self.tau = get_common_hparams(trial, "tau")
        self.n_step = get_common_hparams(trial, "n_step")
        self.start_timesteps = get_common_hparams(trial, "start_timesteps")
        return {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha": self.alpha,
            "tau": self.tau,
            "n_step": self.n_step,
            "start_timesteps": self.start_timesteps,
        }


class TD3HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("td3",)
    _policy_hparams = (
        "actor_lr",
        "critic_lr",
        "n_step",
        "exploration_noise",
        "tau",
        "start_timesteps",
        "update_actor_freq",
        "policy_noise",
        "noise_clip",
    )

    def define_policy_hparams(self, trial: optuna.trial.Trial):
        self.actor_lr = get_common_hparams(trial, "actor_lr")
        self.critic_lr = get_common_hparams(trial, "critic_lr")
        self.exploration_noise = get_common_hparams(trial, "exploration_noise")
        self.tau = get_common_hparams(trial, "tau")
        self.n_step = get_common_hparams(trial, "n_step")
        self.start_timesteps = get_common_hparams(trial, "start_timesteps")
        self.update_actor_freq = get_common_hparams(trial, "update_actor_freq")

        self.noise_clip = 0.5
        self.policy_noise = 0.2
        trial.set_user_attr("policy_noise", self.policy_noise)
        trial.set_user_attr("noise_clip", self.noise_clip)

        return {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "exploration_noise": self.exploration_noise,
            "tau": self.tau,
            "n_step": self.n_step,
            "start_timesteps": self.start_timesteps,
            "update_actor_freq": self.update_actor_freq,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
        }