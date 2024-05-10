import pytest
import gymnasium as gym
import DTRGym  # Assuming this import is necessary for environment registration

# Assuming `DTRGym.registered_ids` gives you a list of all environment IDs
env_ids = DTRGym.registered_ids

@pytest.mark.parametrize("env_id", env_ids)
def test_envs(env_id):
    env = gym.make(env_id)
    env.reset()
    for _ in range(10):
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Env: {env_id}, action: {action}, obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
        env.reset()
    env.close()
