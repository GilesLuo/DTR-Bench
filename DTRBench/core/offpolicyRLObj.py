import os
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy, \
    TD3Policy, SACPolicy, REDQPolicy, C51Policy, DiscreteSACPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import EnsembleLinear
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.utils.net.discrete import Actor as discreteActor
from tianshou.utils.net.discrete import Critic as discreteCritic
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
import gymnasium as gym

from DTRBench.core.base_obj import RLObjective
from DTRBench.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.utils.network import define_single_network, Net, QRDQN, RecurrentPreprocess, Recurrent


class C51Objective(RLObjective):
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device,
                 **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma,
                      lr,
                      stack_num,
                      linear,
                      cat_num,

                      # c51 hp
                      num_atoms,
                      v_min,
                      v_max,
                      estimation_step,
                      target_update_freq,
                      **kwargs):
        # define model
        if stack_num > 1:
            net = Recurrent(layer_num=4, state_shape=self.state_shape, action_shape=self.action_shape,
                            device=self.device, hidden_layer_size=256, num_atoms=num_atoms, last_step_only=True).to(self.device)
        else:
            net = Net(
                self.state_shape,
                self.action_shape,
                hidden_sizes=[256, 256, 256, 256] if not linear else [],
                device=self.device,
                softmax=True,
                num_atoms=num_atoms,
            )

        optim = torch.optim.Adam(net.parameters(), lr=lr)
        policy = C51Policy(model=net,
                           optim=optim,
                           discount_factor=gamma,
                           num_atoms=num_atoms,
                           v_min=v_min,
                           estimation_step=estimation_step,
                           target_update_freq=target_update_freq).to(self.device)
        return policy

    def run(self,
            policy,
            eps_test,
            eps_train,
            eps_train_final,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            **kwargs
            ):

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 1M steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95 * \
                      (eps_train - eps_train_final)
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # seed
        np.random.seed(self.meta_param["seed"])
        torch.manual_seed(self.meta_param["seed"])

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
                stack_num=stack_num
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"],
                                  ignore_obs_next=False,
                                  save_only_last_obs=False,
                                  stack_num=stack_num
                                  )

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=batch_size * self.meta_param["training_num"])
        # trainer

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        )
        return result


class DQNObjective(RLObjective):
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma,
                      lr,
                      stack_num,
                      linear,
                      cat_num,

                      # dqn hp
                      n_step,
                      target_update_freq,
                      is_double,
                      use_dueling,
                      icm_lr_scale=0,  # help="use intrinsic curiosity module with this lr scale"
                      icm_reward_scale=0,  # help="scaling factor for intrinsic curiosity reward"
                      icm_forward_loss_weight=0,  # help="weight for the forward model loss in ICM",
                      **kwargs
                      ):
        # define model
        net = define_single_network(self.state_shape, self.action_shape, use_dueling=use_dueling,
                                    use_rnn=stack_num > 1, device=self.device, linear=linear, cat_num=cat_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = DQNPolicy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,  # we will have a separate runner for double dqn
        )
        if icm_lr_scale > 0:
            feature_net = define_single_network(self.state_shape, 256, use_rnn=False, device=self.device)
            action_dim = int(np.prod(self.action_shape))
            feature_dim = feature_net.output_dim
            icm_net = IntrinsicCuriosityModule(
                feature_net.net,
                feature_dim,
                action_dim,
                hidden_sizes=[512],
                device=self.device
            )
            icm_optim = torch.optim.Adam(icm_net.parameters(), lr=lr)
            policy = ICMPolicy(
                policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
                icm_forward_loss_weight
            ).to(self.device)
        return policy

    def run(self, policy,
            eps_test,
            eps_train,
            eps_train_final,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            **kwargs
            ):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 10k steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * \
                      (eps_train - eps_train_final)
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # seed
        np.random.seed(self.meta_param["seed"])
        torch.manual_seed(self.meta_param["seed"])

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
                stack_num=stack_num
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"],
                                  ignore_obs_next=False,
                                  save_only_last_obs=False,
                                  stack_num=stack_num
                                  )

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        # test train_collector and start filling replay buffer
        print("warm start replay buffer, this may take a while...")
        train_collector.collect(n_step=batch_size * self.meta_param["training_num"])
        # trainer

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        )
        return result


class DDPGObjective(RLObjective):
    # todo: linear does not work
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device,
                 **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self, gamma,
                      actor_lr,
                      critic_lr,
                      stack_num,
                      linear,
                      cat_num,

                      # dqn hp
                      n_step,
                      exploration_noise,
                      tau,
                      **kwargs, ):
        max_action = self.env.action_space.high[0]
        hidden_sizes = [256, 256, 256, 256] if not linear else []
        # model
        net_a = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
        actor = Actor(
            net_a, self.action_shape, max_action=max_action, device=self.device
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
        )
        critic = Critic(net_c, device=self.device).to(self.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        policy = DDPGPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            estimation_step=n_step,
            action_space=self.env.action_space,
        )
        return policy

    def run(self, policy,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            tau,
            exploration_noise,
            start_timesteps,
            **kwargs):
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)

        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.train_envs)
        if start_timesteps > 0:
            train_collector.collect(n_step=start_timesteps, random=True)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            stop_fn=self.early_stop_fn,
            save_checkpoint_fn=self.save_checkpoint_fn
        )
        return result


class REDQObjective(RLObjective):
    # todo: linear does not work
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device,
                 **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self, gamma,
                      actor_lr,
                      critic_lr,
                      stack_num,
                      linear,
                      cat_num,

                      # redq hp
                      ensemble_size,
                      subset_size,
                      tau,
                      alpha,
                      actor_delay,
                      exploration_noise,
                      **kwargs):
        def linear(x, y):
            return EnsembleLinear(ensemble_size, x, y)

        hidden_sizes = [256, 256, 256, 256] if not linear else []
        # model
        net_a = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
        actor = ActorProb(
            net_a, self.action_shape, unbounded=True,
            conditioned_sigma=True, device=self.device
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
        )
        critic = Critic(net_c, device=self.device, flatten_input=False).to(self.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        policy = REDQPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            ensemble_size=ensemble_size,
            subset_size=subset_size,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            actor_delay=actor_delay,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            action_space=self.env.action_space,
        )
        return policy

    def run(self, policy,
            step_per_collect,
            update_per_step,
            stack_num,
            cat_num,
            batch_size,
            tau,
            exploration_noise,
            start_timesteps,
            **kwargs):
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.train_envs)
        if start_timesteps > 0:
            train_collector.collect(n_step=start_timesteps, random=True)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            stop_fn=self.early_stop_fn,
            save_checkpoint_fn=self.save_checkpoint_fn
        )
        return result


class SACObjective(DDPGObjective):
    # todo: linear does not work
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device,
                 **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self, gamma,
                      stack_num,
                      actor_lr,
                      critic_lr,
                      alpha,
                      n_step,
                      tau,
                      cat_num,
                      linear,
                      **kwargs, ):
        hidden_sizes = [256, 256, 256] if not linear else []

        # model
        net_a = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
        actor = ActorProb(
            net_a,
            self.action_shape,
            device=self.device,
            unbounded=True,
            conditioned_sigma=True,
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c1 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        net_c2 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        critic1 = Critic(net_c1, device=self.device).to(self.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=self.device).to(self.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            estimation_step=n_step,
            action_space=self.action_space,
        )
        return policy

    def run(self, policy,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            start_timesteps,
            **kwargs):
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.train_envs)
        if start_timesteps > 0:
            train_collector.collect(n_step=start_timesteps, random=True)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            stop_fn=self.early_stop_fn,
            save_checkpoint_fn=self.save_checkpoint_fn
        )
        return result


class DiscreteSACObjective(SACObjective):
    # todo: linear does not work
    def define_policy(self, gamma,
                      stack_num,
                      actor_lr,
                      critic_lr,
                      alpha,
                      n_step,
                      tau,
                      cat_num,
                      linear,
                      **kwargs, ):
        hidden_sizes = [256, 256, 256] if not linear else ()  # actor and critic has another layer
        if stack_num > 1:
            net_actor = RecurrentPreprocess(layer_num=3, state_shape=self.state_shape,
                                            device=self.device, hidden_layer_size=256).to(self.device)
            net_critic1 = RecurrentPreprocess(layer_num=3, state_shape=self.state_shape,
                                              device=self.device, hidden_layer_size=256).to(self.device)
            net_critic2 = RecurrentPreprocess(layer_num=3, state_shape=self.state_shape,
                                              device=self.device, hidden_layer_size=256).to(self.device)
        else:
            net_actor = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
            net_critic1 = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
            net_critic2 = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
        actor = discreteActor(
            net_actor, self.action_shape, device=self.device, softmax_output=False, preprocess_net_output_dim=256).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic1 = discreteCritic(net_critic1, last_size=self.action_shape, device=self.device,
                                 preprocess_net_output_dim=256).to(self.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = discreteCritic(net_critic2, last_size=self.action_shape, device=self.device,
                                 preprocess_net_output_dim=256).to(self.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            estimation_step=n_step,
            action_space=self.action_space,
        )
        return policy


class TD3Objective(RLObjective):
    # todo: linear does not work
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self, gamma,
                      actor_lr,
                      critic_lr,
                      n_step,
                      tau,
                      update_actor_freq,
                      policy_noise,
                      noise_clip,
                      exploration_noise,
                      cat_num,
                      linear,
                      **kwargs, ):

        # model
        max_action = self.action_space.high[0]
        hidden_sizes = [256, 256, 256, 256] if not linear else []
        net_a = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
        actor = Actor(
            net_a, self.action_shape, max_action=max_action, device=self.device
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c1 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        net_c2 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        critic1 = Critic(net_c1, device=self.device).to(self.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=self.device).to(self.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        policy = TD3Policy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            estimation_step=n_step,
            action_space=self.action_space,
        )
        return policy

    def run(self, policy,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            start_timesteps,
            **kwargs):
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.train_envs)
        if start_timesteps > 0:
            train_collector.collect(n_step=start_timesteps, random=True)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            stop_fn=self.early_stop_fn,
            save_checkpoint_fn=self.save_checkpoint_fn
        )
        return result
