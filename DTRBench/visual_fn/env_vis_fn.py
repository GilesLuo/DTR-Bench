import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tianshou.data import Batch
from tqdm import tqdm
import os
from typing import Tuple


def get_one_episode_data(env, policy, seed) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    observation, info = env.reset(seed=seed)
    done = False
    step = 0

    states = Batch({key: np.full((env.unwrapped.max_t + 1,), np.nan) for key in env.unwrapped.env_info["state_key"]})
    observations = np.full((env.unwrapped.max_t+1, env.observation_space.shape[0]), np.nan)
    rewards = np.full((env.unwrapped.max_t + 1, 2), np.nan)
    if info["action"] is None:
        action_shape = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
    else:
        action_shape = len(info["action"])
    actions = np.full((env.unwrapped.max_t + 1, action_shape), np.nan)

    states[step] = {key: value for key, value in info["state"].items() if key in env.unwrapped.env_info["state_key"]}
    observations[step] = observation
    rewards[step, 0] = 0
    rewards[step, 1] = 0  # to align with the length of obs for plotting
    actions[step] = 0 if action_shape == 1 else np.zeros(action_shape)

    hidden_state = Batch(hidden=torch.zeros((1, 3, 256)), cell=torch.zeros((1, 3, 256)))
    while not done:
        step += 1
        out = policy.forward(batch=Batch(obs=np.expand_dims(observation, axis=0), info=info), state=hidden_state)
        action = out.act[0]
        if type(action) == torch.Tensor: # for the continuous policy
            action = action.detach().cpu().numpy()
        if isinstance(env.action_space, gym.spaces.Box):
            act_high = env.action_space.high
            act_low = env.action_space.low
            action = act_low + (action / act_high + 1.0) * 0.5 * (act_high - act_low)
        hidden_state = out.state
        observation, reward, terminated, truncated, info = env.step(action)
        states[step] = {key: value for key, value in info["state"].items() if key in env.env_info["state_key"]}
        observations[step] = observation
        rewards[step, 0] = reward
        rewards[step, 1] = info["instantaneous_reward"]
        if "action" in info:
            actions[step, :] = info["action"]  # some envs have a different action format than the one given to the policy
        else:
            actions[step] = action

        done = terminated or truncated
    return states, observations, rewards, actions, step


def visualise_env_policy_fn(env, policy, plot_save_path=None, confidence=0.95, num_episodes=1000,
                            fig_size=(10, 10), use_log_scale=False, seed=None, drop_last_step=False):
    unwrapped_env = env.unwrapped
    states = Batch({key: np.full((num_episodes, unwrapped_env.max_t + 1,), np.nan) for key in unwrapped_env.env_info["state_key"]})
    observations = np.full((num_episodes, unwrapped_env.max_t + 1, unwrapped_env.observation_space.shape[0]), np.nan)
    rewards = np.full((num_episodes, unwrapped_env.max_t + 1, 2), np.nan)

    seeds = seed or np.random.randint(0, 100000, num_episodes)
    step_nums = np.full((num_episodes,), np.nan)
    obs_colors = ['r', 'b', 'g', 'y', 'k', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                  'lime']
    act_colors = ['navy', 'teal', 'maroon', 'gold', 'silver', 'bronze', 'turquoise', 'indigo', 'beige', 'khaki',
                  'coral', 'sage', 'salmon', 'chartreuse', 'lavender']
    reward_colors = ['chocolate', 'plum', 'ivory', 'mint', 'pearl', 'peach', 'amethyst', 'ebony', 'jade',
                     'ruby', 'sapphire', 'topaz', 'amber', 'emerald']

    # collecting episodes data
    for i in tqdm(range(num_episodes), desc=f"{policy.__class__.__name__} testing episodes"):
        seed = int(seeds[i])
        state, obs, reward, action, step_num = get_one_episode_data(env, policy, seed)
         # unwrap after episode collection
        states[i] = state
        observations[i] = obs
        rewards[i] = reward
        if i == 0:
            action_shape = action.shape[1]
            actions = np.full((num_episodes, unwrapped_env.max_t + 1, action_shape), np.nan)
        actions[i] = action
        step_nums[i] = step_num

    num_plots = 4  # one for state, one for observation, one for action, one for reward
    obs_key = unwrapped_env.env_info["obs_key"]
    act_key = unwrapped_env.env_info["act_key"]
    fig, axes = plt.subplots(num_plots, figsize=fig_size)

    for i, ax in enumerate(axes):  # plot obs and reward
        if i == 0:  # state plot
            obs_color_dict = {}
            for j, (key, state_episode) in enumerate(states.items()):
                obs_color_dict[key] = obs_colors[j]
                mean_values, l_bound, u_bound = cal_mean_std(state_episode, confidence=confidence)
                if drop_last_step:
                    mean_values = mean_values[:-1]
                    l_bound = l_bound[:-1]
                    u_bound = u_bound[:-1]
                ax.plot(range(len(mean_values)), mean_values, label=key, color=obs_color_dict[key])
                ax.fill_between(
                    range(len(mean_values)),
                    l_bound,
                    u_bound,
                    color=obs_color_dict[key],
                    alpha=.2,
                )
            ax.set_ylabel('States')
            ax.legend(loc='upper right')
            if use_log_scale:
                ax.set_yscale('log')
        elif i == 1:  # observation plot
            for j, key in enumerate(obs_key):
                color = obs_color_dict[key] if key in obs_color_dict.keys() else obs_colors[j]
                obs_episode = observations[:, :, j]
                mean_values, l_bound, u_bound = cal_mean_std(obs_episode, confidence=confidence)
                if drop_last_step:
                    mean_values = mean_values[:-1]
                    l_bound = l_bound[:-1]
                    u_bound = u_bound[:-1]
                ax.plot(range(len(mean_values)), mean_values, label=key, color=color)
                ax.fill_between(
                    range(len(mean_values)),
                    l_bound, u_bound,
                    color=color,
                    alpha=.2,
                )
            ax.set_ylabel('Observations')
            ax.legend(loc='upper right')
            if use_log_scale:
                ax.set_yscale('log')
        elif i == 2:  # action plot
            for j in range(action_shape):
                act_episode = actions[:, :, j]
                mean_values, l_bound, u_bound = cal_mean_std(act_episode, confidence=confidence)
                if drop_last_step:
                    mean_values = mean_values[:-1]
                    l_bound = l_bound[:-1]
                    u_bound = u_bound[:-1]
                ax.plot(range(len(mean_values)), mean_values, label=act_key[j], color=act_colors[j])
                ax.fill_between(
                    range(len(mean_values)),
                    l_bound,
                    u_bound,
                    color=act_colors[j],
                    alpha=.2,
                    )
                ax.set_ylabel('Action')
                ax.legend(loc='upper right')
        elif i == 3:  # reward plot
            reward_labels = ['reward', 'instantaneous_reward']
            for j in range(rewards.shape[-1]):
                if j != 0:
                    break
                reward_episode = rewards[:, :, j]
                mean_values, l_bound, u_bound = cal_mean_std(reward_episode, confidence=confidence)
                if drop_last_step:
                    mean_values = mean_values[:-1]
                    l_bound = l_bound[:-1]
                    u_bound = u_bound[:-1]
                ax.plot(range(len(mean_values)), mean_values, label=reward_labels[j], color=reward_colors[j])
                ax.fill_between(
                    range(len(mean_values)),
                    l_bound, u_bound,
                    color=reward_colors[j],
                    alpha=.2,
                    )
                ax.set_xlabel('Step')
                ax.set_ylabel('Reward')
                ax.legend(loc='upper right')


    if plot_save_path is not None:
        plot_save_dir, _ = os.path.split(plot_save_path)
        Path(plot_save_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_save_path)
    else:
        plt.show()


def cal_confidence_interval(data: np.array, n_iterations=None, confidence_level=0.95):
    # Set the number of bootstrap iterations
    if not n_iterations:
        n_iterations = len(data)

    # Create an array to store the bootstrap sample means
    bootstrap_means = np.empty(n_iterations)

    # Perform bootstrapping
    for i in range(n_iterations):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Calculate the lower and upper percentiles of the bootstrap means
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile

    lower_bound = np.percentile(bootstrap_means, lower_percentile * 100)
    upper_bound = np.percentile(bootstrap_means, upper_percentile * 100)
    return bootstrap_means, lower_bound, upper_bound


def cal_mean_std(data: np.array, **kwargs):
    # data shape [n_episodes, max_step]
    if data.shape[0] > 1:
        mean_values = np.nanmean(data, axis=0)
        std_values = np.nanstd(data, axis=0)
    else:
        mean_values = data[0]
        std_values = np.zeros_like(mean_values)
    mean_values = mean_values[~np.isnan(mean_values)]
    std_values = std_values[:len(mean_values)]
    return mean_values, mean_values + std_values, mean_values - std_values

