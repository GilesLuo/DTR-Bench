import os

from gymnasium import spaces
from gymnasium import Wrapper
import numpy as np
import warnings


class DiscreteActionWrapper(Wrapper):
    def __init__(self, env, n_act: int):
        super().__init__(env)
        assert isinstance(env.action_space,
                          spaces.Box), "This wrapper only works with continuous action spaces (gym.spaces.Dict)"
        # Change the action space to be discrete
        self.n_act = n_act
        self.action_space = spaces.Discrete(self.n_act)
        self.env_info["action_type"] = "discrete"

    def reset(self, seed: int = None, **kwargs):
        return self.env.reset(seed, **kwargs)

    def step(self, action: int):
        act_bin = np.linspace(self.env.action_space.low, self.env.action_space.high, self.n_act)
        return self.env.step(np.array(act_bin[action]))


def is_square_number(n):
    if n < 0:
        return False
    for i in range(1, n // 2):
        if i * i == n:
            return True
        if i * i > n:
            return False


class DiscreteDoubleActionWrapper(DiscreteActionWrapper):
    def __init__(self, env, n_act: int = 25):
        assert is_square_number(n_act), "please make sure n_act is a square number"
        super().__init__(env=env, n_act=n_act)
        self.action_space = spaces.Discrete(self.n_act)
        if np.sqrt(self.n_act) != int(np.sqrt(self.n_act)):
            warnings.warn("n_act is not a square number, the action space is not a square matrix, "
                          f"n_act will be rounded to the nearest square number {int(np.sqrt(self.n_act))} * {int(np.sqrt(self.n_act))}")

    def step(self, action: np.array):
        if action is not np.array:
            action = np.array(action)
        arr_matrix = np.arange(self.n_act).reshape(int(np.sqrt(self.n_act)), int(np.sqrt(self.n_act)))
        act_index = np.where(arr_matrix == action)

        act_bin = np.linspace(self.env.action_space.low, self.env.action_space.high, int(np.sqrt(self.n_act)))
        continuous_action = np.zeros(shape=self.env.action_space.shape)
        for i in range(len(act_index)):
            continuous_action[i] = act_bin[int(act_index[i])][i]
        return self.env.step(continuous_action.astype(np.float32))


from copy import deepcopy
from numba import jit


def get_neighbors(matrix: np.ndarray, indices: tuple, radius=1):
    assert len(matrix.shape) == 2, "please enter a matrix"
    assert indices[0] < matrix.shape[0] and indices[1] < matrix.shape[1], "please give valid indices"
    point_value = matrix[indices[0], indices[1]]

    row_start_indice, row_end_indice = max(
        0, indices[0] - radius), min(matrix.shape[0], indices[0] + radius + 1)
    column_start_indice, column_end_indice = max(
        0, indices[1] - radius), min(matrix.shape[1], indices[1] + radius + 1)
    neighbors = deepcopy(matrix[row_start_indice: row_end_indice,
                         column_start_indice: column_end_indice])

    neighbors[indices[0] - row_start_indice, indices[1] - column_start_indice] = np.nan
    neighbors = list(neighbors.reshape(-1))
    neighbors = [e for e in neighbors if not np.isnan(e)]
    return neighbors


def get_neighbors_indices(matrix: np.ndarray, indices: tuple, radius=1):
    assert len(matrix.shape) == 2, "please enter a matrix"
    assert indices[0] < matrix.shape[0] and indices[1] < matrix.shape[1], "please give valid indices"

    row_start_indice, row_end_indice = max(
        0, indices[0] - radius), min(matrix.shape[0], indices[0] + radius + 1)
    column_start_indice, column_end_indice = max(
        0, indices[1] - radius), min(matrix.shape[1], indices[1] + radius + 1)

    neighbor_indices = []
    for i in range(row_start_indice, row_end_indice):
        for j in range(column_start_indice, column_end_indice):
            if i != indices[0] or j != indices[1]:
                neighbor_indices.append((i, j))

    return neighbor_indices




