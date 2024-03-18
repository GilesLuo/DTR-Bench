import numpy as np
import os
from tianshou.data import Batch
import gymnasium as gym
from pathlib import Path
import pandas as pd


class BasePolicy():
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        if isinstance(self.action_space, gym.spaces.Box):
            self.act_max = self.action_space.high
            self.act_min = self.action_space.low
            self.act_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.act_shape = (1,)
            self.act_max = self.action_space.n
            self.act_min = 0

    def forward(self, batch: Batch, state=None):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RandomPolicy(BasePolicy):
    def forward(self, batch: Batch, state=None):
        batch_size = batch.obs.shape[0]

        act = np.random.rand(batch_size, *self.act_shape)
        act = act * (self.act_max - self.act_min) + self.act_min
        return Batch(act=act, state=None)



class MinPolicy(BasePolicy):
    def forward(self, batch: Batch, state=None):
        batch_size = batch.obs.shape[0]
        act = np.tile(self.act_min, (batch_size, 1))
        return Batch(act=act, state=np.nan)


class MaxPolicy(BasePolicy):
    def forward(self, batch: Batch, state=None):
        batch_size = batch.obs.shape[0]
        act = np.tile(self.act_max, (batch_size, 1))
        return Batch(act=act, state=np.nan)


def eval_baseline_policy(policy_name, env_name, log_dir: str,
                         test_num=5000, num_seed=5):
    """
    Evaluate a baseline policy (no training required) and return the test results.

    Parameters:
    - policy_name: Name of the policy to be tested (RandomPolicy, MinPolicy, MaxPolicy)
    - log_dir: Directory to save the test results
    - env_name: Environment name, a string used to create a gym environment
    - test_num: Total number of episodes for the test
    - num_seed: number of different random seeds to use for testing

    Returns:
    - A dictionary containing reward mean, reward standard deviation, length mean, and length standard deviation
    """
    env = gym.make(env_name)
    if policy_name == "random":
        policy = RandomPolicy(action_space=env.action_space)
    elif policy_name == "min":
        policy = MinPolicy(action_space=env.action_space)
    elif policy_name == "max":
        policy = MaxPolicy(action_space=env.action_space)
    else:
        raise ValueError(f"Invalid policy name: {policy_name}")

    np.random.seed(0)
    seeds = np.random.randint(0, 100000, num_seed)
    total_rewards_list = []
    total_lengths_list = []
    for seed in seeds:
        seed = int(seed)
        env.seed(seed)
        np.random.seed(seed)
        rewards_list = []
        lengths_list = []
        test_result_fname = f"{env_name}-{policy_name}-{seed}"
        test_result_save_dir = os.path.join(log_dir, env_name, f"{policy_name}-best", str(seed))
        Path(test_result_save_dir).mkdir(parents=True, exist_ok=True)

        for _ in range(test_num):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            length = 0

            while not done:
                # Wrap observation in batch format to simulate batched environment
                batch_obs = np.expand_dims(obs, axis=0)
                action = policy(Batch(obs=batch_obs)).act[0]  # Retrieve action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                length += 1

            rewards_list.append(total_reward)
            lengths_list.append(length)

        total_rewards_list.extend(rewards_list)
        total_lengths_list.extend(lengths_list)
        test_result = {"rews": rewards_list,
                       "lens": lengths_list}

        test_df = pd.DataFrame(test_result, index=[0])
        test_df.to_csv(os.path.join(test_result_save_dir, f"{test_result_fname}.csv"), index=False)

    test_result_stats = {"rew_mean": np.mean(total_rewards_list),
                         "rew_std": np.std(total_rewards_list),
                         "len_mean": np.mean(total_lengths_list),
                         "len_std": np.std(total_lengths_list)}
    return test_result_stats


def demo_run_policy(policy, action_space, batch_size=10):
    """
    Test a given policy with a mock action space and batch size.

    Parameters:
    - policy: The policy to be tested (RandomPolicy, MinPolicy, MaxPolicy).
    - action_space: An instance of gym.spaces that defines the action space.
    - batch_size: The number of observations in the batch to test the policy.
    """

    # Mock a batch of observations (random for simplicity)
    obs = np.random.randn(batch_size, *action_space.shape)\
                          if hasattr(action_space, 'shape')\
                          else np.random.randint(0, action_space.n, size=(batch_size, 1))
    batch = Batch(obs=obs)

    # Use the policy to generate actions for the batch
    output_batch = policy(batch)

    # Display the generated actions (and state, if applicable)
    print(f"Actions generated by {policy.__class__.__name__}:")
    print(output_batch.act)
    if output_batch.state is not None:
        print("State:", output_batch.state)
    else:
        print("No state information.")


# Example usage
if __name__ == "__main__":
    # Define a continuous action space
    action_space_box = gym.spaces.Box(low=np.array([-1.0, -1.0]),
                                      high=np.array([1.0, 1.0]),
                                      dtype=np.float32)

    # Define a discrete action space
    action_space_discrete = gym.spaces.Discrete(5)

    # Test the policies with a continuous action space
    print("Testing with a continuous action space:")
    demo_run_policy(RandomPolicy(action_space_box), action_space_box)
    demo_run_policy(MinPolicy(action_space_box), action_space_box)
    demo_run_policy(MaxPolicy(action_space_box), action_space_box)

    # Test the policies with a discrete action space
    print("\nTesting with a discrete action space:")
    demo_run_policy(RandomPolicy(action_space_discrete), action_space_discrete)
    demo_run_policy(MinPolicy(action_space_discrete), action_space_discrete)
    demo_run_policy(MaxPolicy(action_space_discrete), action_space_discrete)
