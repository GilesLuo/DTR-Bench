import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy.base import BasePolicy
from tianshou.utils import TensorboardLogger, WandbLogger
from torch.utils.tensorboard import SummaryWriter

from DTRGym import buffer_registry
from DTRGym.base import make_env
from DTRBench.utils.misc import set_global_seed
from DTRBench.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.utils.data import load_buffer


class RLObjective:
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace,
                 device, logger="tensorboard", multi_obj=False, early_stopping=10, **kwargs
                 ):

        # define high level parameters
        self.env_name, self.hparam_space, self.logger_type, self.early_stopping = env_name, hparam_space, logger, early_stopping
        self.logger = None
        self.device = device
        self.multi_obj = multi_obj
        self.meta_param = self.hparam_space.get_meta_params()
        self.retrain = False  # enable checkpoint saving in retrain otherwise disable

        # define job name for logging
        self.job_name = self.env_name

        # early stopping counter
        self.rew_history = []
        self.early_stopping_counter = 0

        # prepare env
        self.env, self.train_envs, self.test_envs = make_env(env_name, self.meta_param["seed"],
                                                             self.meta_param["training_num"],
                                                             1,
                                                             # test_num is always 1, we will run test multiple times for one test env
                                                             num_actions=self.meta_param["num_actions"])

        state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.state_space = self.env.observation_space
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.action_space = self.env.action_space
        if isinstance(state_shape, (tuple, list)):
            if len(state_shape) > 1:
                raise NotImplementedError("state shape > 1 not supported yet")
            self.state_shape = state_shape[0]
        else:
            self.state_shape = int(state_shape)
        if isinstance(action_shape, (tuple, list)):
            if len(action_shape) > 1:
                raise NotImplementedError("action shape > 1 not supported yet")
            self.action_shape = action_shape
        else:
            self.action_shape = int(action_shape)

    def __call__(self, trial):
        self.retrain = False
        hparams = self.hparam_space(trial)
        set_global_seed(self.meta_param["seed"])

        # define logger
        log_name = os.path.join(self.job_name, self.meta_param["algo_name"],
                                f"trial{trial.number}-seed{self.meta_param['seed']}")
        log_path = os.path.join(self.meta_param["logdir"], log_name)
        if self.logger_type == "wandb":
            self.logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "_"),
                run_id=None,
                config=self.hparam_space.get_all_params(),
                project="DTRBench",
            )

        writer = SummaryWriter(log_path)
        writer.add_text("args", str(self.hparam_space.get_all_params()))
        if self.logger_type == "tensorboard":
            self.logger = TensorboardLogger(writer, update_interval=10)
        else:  # wandb
            self.logger.load(writer)
        self.log_path = log_path

        self.policy = self.define_policy(**hparams)

        result = self.run(self.policy, **hparams)
        _, best_env_step = self.get_info_from_logger()
        score = result['best_reward']

        for k, v in result.items():
            trial.set_user_attr(f"eval:{k}", v)
        trial.set_user_attr(f"eval:best_step", best_env_step)

        if self.multi_obj:
            return score, best_env_step
        else:
            return score

    def get_info_from_logger(self):
        if isinstance(self.logger, WandbLogger):
            log_dir = wandb.run.dir
            for f in os.listdir(log_dir):
                if f.startswith("events.out.tfevents"):
                    log_file_path = os.path.join(log_dir, f)
                    break
        elif isinstance(self.logger, TensorboardLogger):
            log_file_path = os.path.join(self.log_path, "events.out.tfevents.*")
        else:
            raise NotImplementedError
        event_acc = EventAccumulator(self.log_path)
        event_acc.Reload()
        reward = event_acc.Scalars('test/reward')

        df = []
        for r in reward:
            df.append([r.step, r.value])
        df = pd.DataFrame(df, columns=['step', 'reward']).sort_values('reward', ascending=False)
        best_reward, env_step = df.iloc[0]['reward'], df.iloc[0]['step']
        return best_reward, env_step

    def early_stop_fn(self, mean_rewards):
        # reach reward threshold
        reach = False
        if self.env.spec.reward_threshold:
            reach = mean_rewards >= self.env.spec.reward_threshold

        # reach early stopping
        early_stop = False
        # todo: early stopping is not working for now, because stop_fn is called at each training step
        # if self.early_stopping > 0:
        # self.rew_history.append(mean_rewards)
        # if max(self.rew_history) >mean_rewards:
        #     self.early_stopping_counter += 1
        # else:
        #     self.early_stopping_counter = 0
        # early_stop = self.early_stopping_counter >= self.early_stopping
        # print("training stop at epoch {} due to early stopping, "
        #       "last epoch reward: {}, best reward: {}".format(len(self.rew_history), mean_rewards, max(self.rew_history)))
        return reach or early_stop

    # no checkpoint saving needed
    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # if self.retrain:
        #     ckpt_path = os.path.join(self.log_path, f"checkpoint_{epoch}.pth")
        #     torch.save({"model": self.policy.state_dict()}, ckpt_path)
        #     return ckpt_path
        pass
    
    def define_policy(self, *args, **kwargs) -> BasePolicy:
        return NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def retrain_best_hparam(self, final_test_num:int,
                            num_seed: int = 5,
                            **hparams):
        # define logger
        self.retrain = True
        # rewrite meta_param
        for key in self.meta_param.keys():
            self.meta_param[key] = hparams[key]
        if "seed" in hparams.keys():
            print("Seed will be replaced by separate retrain seeds in retrain_best_hparam")

        np.random.seed(0)
        seeds = np.random.randint(0, 100000, num_seed)
        test_seed = np.random.randint(0, 100000, 1)[0]
        rew_list = []
        len_list = []
        for seed in seeds:
            result_path = os.path.join(self.meta_param["logdir"], self.job_name, f"{self.meta_param['algo_name']}-best",
                                       str(seed), f"{self.job_name}-{self.meta_param['algo_name']}-{seed}.csv")
            self.meta_param["seed"] = seed
            log_name = os.path.join(self.job_name, f"{self.meta_param['algo_name']}-best", str(self.meta_param["seed"]))
            log_path = os.path.join(self.meta_param["logdir"], log_name)
            if not os.path.exists(result_path):
                print("Retrain best hparam with seed {}".format(seed))

                self.env, self.train_envs, self.test_envs = make_env(self.env_name, int(self.meta_param["seed"]),
                                                                     self.meta_param["training_num"],
                                                                     1,
                                                                     num_actions=self.meta_param[
                                                                         "num_actions"])  # to reset seed
                if self.logger_type == "wandb":
                    self.logger = WandbLogger(
                        save_interval=1,
                        name=log_name.replace(os.path.sep, "__"),
                        run_id="0",
                        config=self.hparam_space.get_all_params(),
                        project=self.job_name,
                    )

                writer = SummaryWriter(log_path)
                if self.logger_type == "tensorboard":
                    self.logger = TensorboardLogger(writer, update_interval=10)
                else:  # wandb
                    self.logger.load(writer)
                self.log_path = log_path

                self.policy = self.define_policy(**hparams)
                self.policy.train()
                np.random.seed(self.meta_param["seed"])
                torch.manual_seed(self.meta_param["seed"])

                self.run(self.policy, **hparams)

                # test the policy on a separate seed
                self.policy.load_state_dict(torch.load(os.path.join(self.log_path, "policy.pth")))
                self.policy.eval()
                test_env = DummyVectorEnv([lambda: gym.make(self.env_name, n_act=self.meta_param["num_actions"])])
                test_env.seed(int(test_seed))
                test_collector = Collector(self.policy, test_env, exploration_noise=True)
                result = test_collector.collect(n_episode=final_test_num, render=False)
                result_df = pd.DataFrame({"rews": np.squeeze(result["rews"]), "lens": np.squeeze(result["lens"])})
                result_df.to_csv(result_path)
            else:
                # check if the result length is test_num
                result_df = pd.read_csv(result_path)
                if len(result_df) != final_test_num:
                    print("test num {} not match, retesting with {}".format(len(result_df), final_test_num))
                    # test the policy on a separate seed
                    policy = self.define_policy(**hparams)
                    # check if retraining is done
                    if len([ckpt for ckpt in os.listdir(log_path) if ckpt.endswith(".pth") and "checkpoint" in ckpt]) < \
                            self.meta_param["epoch"]:
                        # remove the csv file and retrain
                        os.remove(result_path)
                        self.retrain_best_hparam(self, final_test_num, **hparams)
                    policy.load_state_dict(torch.load(os.path.join(log_path, "policy.pth")))
                    policy.eval()
                    test_env = DummyVectorEnv([lambda: gym.make(self.env_name, n_act=self.meta_param["num_actions"])])
                    test_env.seed(int(test_seed))
                    test_collector = Collector(policy, test_env, exploration_noise=True)
                    result = test_collector.collect(n_episode=final_test_num, render=False)
                    result_df = pd.DataFrame({"rews": np.squeeze(result["rews"]), "lens": np.squeeze(result["lens"])})
                    result_df.to_csv(result_path)
                else:
                    print("seed {} retrained".format(seed))

            rew_list.extend(result_df["rews"].to_list())
            len_list.extend(result_df["lens"].to_list())

        return {"rew_mean": np.mean(rew_list), "rew_std": np.std(rew_list),
                "len_mean": np.mean(len_list), "len_std": np.std(len_list)}