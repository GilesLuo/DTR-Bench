import torch
import gymnasium as gym
import numpy as np

import DTRGym


class TileCoderTorch:
    def __init__(self, num_tiles: int, num_bins_per_dim: int,
                 observation_space: gym.spaces.Box):
        self.num_bins_per_dim = num_bins_per_dim
        self.num_obs = observation_space.shape[0]
        self.num_tiles = num_tiles

        self.bins = []
        self.offsets = []
        for i_obs in range(self.num_obs):
            low = observation_space.low[i_obs]
            high = observation_space.high[i_obs]
            # Using torch.linspace to create bins and offsets
            self.bins.append(torch.linspace(low, high, num_bins_per_dim + 1)[1:-1])
            self.offsets.append(torch.linspace(low, high / 2, num_tiles))

        # Convert lists to PyTorch tensors
        self.bins = torch.stack(self.bins)  # (n_obs, n_bins)
        self.offsets = torch.stack(self.offsets).t()  # (n_tiles, n_obs)

    def encode(self, state: torch.Tensor, use_onehot: bool = True, use_flatten: bool = True):
        assert state.shape[0] == self.num_obs

        # Pre-allocate tensor for codes
        codes = torch.zeros((self.num_tiles, self.num_obs), dtype=torch.int32)
        for i_tile in range(self.num_tiles):
            for i_obs in range(self.num_obs):
                offset = self.offsets[i_tile, i_obs]
                # Using torch.digitize to find bins
                codes[i_tile, i_obs] = torch.bucketize(state[i_obs], self.bins[i_obs] + offset, right=True)

        if use_onehot:
            # One-hot representation using PyTorch operations
            state_repr = torch.zeros((self.num_tiles, self.num_obs, self.num_bins_per_dim), dtype=torch.float32)
            for i_tile in range(self.num_tiles):
                for i_obs in range(self.num_obs):
                    bin_idx = codes[i_tile, i_obs]
                    state_repr[i_tile, i_obs, bin_idx] = 1.0
        else:
            state_repr = codes

        if use_flatten:
            state_repr = state_repr.flatten()

        return state_repr


class TileCoderNumpy:
    def __init__(self, num_tiles: int, num_bins_per_dim: int,
                 observation_space: gym.spaces.Box):
        self.num_bins_per_dim = num_bins_per_dim
        self.num_obs = observation_space.shape[0]
        self.num_tiles = num_tiles

        self.bins = []
        self.offsets = []
        for i_obs in range(self.num_obs):
            low = observation_space.low[i_obs]
            high = observation_space.high[i_obs]
            self.bins.append(np.linspace(low, high, num_bins_per_dim + 1)[1:-1])
            self.offsets.append(np.linspace(low, high/2, num_tiles))

        self.bins = np.array(self.bins)  # (n_obs, n_bins)
        self.offsets = np.transpose(np.array(self.offsets))  # (n_tiles, n_obs)

    def encode(self, obs: np.array, use_onehot:bool=True, use_flatten:bool=True):
        assert obs.shape[0] == self.num_obs

        codes = np.zeros((self.num_tiles, self.num_obs), dtype=np.int32)
        for i_tile in range(self.num_tiles):
            for i_obs in range(self.num_obs):
                offset = self.offsets[i_tile, i_obs]
                codes[i_tile, i_obs] = np.digitize(obs[i_obs], self.bins[i_obs]+offset)

        if use_onehot:
            # Convert to one-hot representation for linear function approximation
            obs_repr = np.zeros((self.num_tiles, self.num_obs, self.num_bins_per_dim), dtype=np.int32)
            for i_tile in range(self.num_tiles):
                for i_obs in range(self.num_obs):
                    bin_idx = codes[i_tile, i_obs]
                    obs_repr[i_tile, i_obs, bin_idx] = 1
        else:
            obs_repr = codes

        if use_flatten:
            obs_repr = obs_repr.reshape(-1)

        return obs_repr


def demo_TileCoderTorch():
    env = gym.make('AhnChemoEnv-continuous')
    observation_space = env.observation_space

    # Create an instance of the TileCoder
    num_tiles = 8  # number of tilings
    num_bins_per_dim = 10  # number of bins per dimension for each tiling
    tile_coder = TileCoderNumpy(num_tiles, num_bins_per_dim, observation_space)

    obs = env.reset()
    encoded_obs = tile_coder.encode(torch.Tensor(obs[0]))
    print("Encoded obs (one-hot representation):\n", encoded_obs)

    flattened_obs = tile_coder.encode(torch.Tensor(obs[0]), use_flatten=True)
    print("\nFlattened obs representation:\n", flattened_obs)

    env.close()


def demo_TileCoderNumpy():
    env = gym.make('AhnChemoEnv-continuous')
    observation_space = env.observation_space

    # Create an instance of the TileCoder
    num_tiles = 8  # number of tilings
    num_bins_per_dim = 10  # number of bins per dimension for each tiling
    tile_coder = TileCoderNumpy(num_tiles, num_bins_per_dim, observation_space)

    obs = env.reset()
    encoded_obs = tile_coder.encode(obs[0])
    print("Encoded obs (one-hot representation):\n", encoded_obs)

    flattened_obs = tile_coder.encode(obs[0], use_flatten=True)
    print("\nFlattened obs representation:\n", flattened_obs)

    env.close()


if __name__ == "__main__":
    demo_TileCoderTorch()
    # demo_TileCoderNumpy()

