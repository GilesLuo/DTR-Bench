from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy
from DTRBench.utils.tile_coding import TileCoderTorch
import copy


class RuleBasedPolicy(BasePolicy):
    def __init__(self, action_space, *args, **kwargs):
        super().__init__()
        self.action_space = action_space
        self.check_action_space()

    def check_action_space(self):
        if not isinstance(self.action_space, gym.spaces.Space):
            raise TypeError(f"RuleBasedPolicy.action_space must be gym.spaces.Space, now {type(self.action_space)}")

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class RandomPolicy(RuleBasedPolicy):
    """Choose an action randomly."""

    def forward(self, batch: Batch, state=None, **kwargs):

        # If the action space is discrete
        if isinstance(self.action_space, Discrete):
            actions = np.random.randint(0, self.action_space.n, size=(len(batch.obs),))

        # If the action space is continuous
        elif isinstance(self.action_space, Box):
            low = self.action_space.low
            high = self.action_space.high
            actions = np.random.uniform(low, high, size=(len(batch.obs),) + low.shape)

        else:
            raise ValueError("Unsupported action space type.")

        return Batch(act=actions, state=state)


class AllMinPolicy(RuleBasedPolicy):
    """Always choose the action with index 0 or the minimum value."""

    def forward(self, batch: Batch, state=None, **kwargs):
        if isinstance(self.action_space, Discrete):
            actions = np.zeros((len(batch.obs),), dtype=int)
        elif isinstance(self.action_space, Box):
            actions = np.tile(self.action_space.low, (len(batch.obs), 1))
        else:
            raise ValueError("Unsupported action space type.")
        return Batch(act=actions, state=state)


class AllMaxPolicy(RuleBasedPolicy):
    """Always choose the action with the highest index or the maximum value."""

    def forward(self, batch: Batch, state=None, **kwargs):

        if isinstance(self.action_space, Discrete):
            actions = np.full((len(batch.obs),), self.action_space.n - 1, dtype=int)
        elif isinstance(self.action_space, Box):
            actions = np.tile(self.action_space.high, (len(batch.obs), 1))
        else:
            raise ValueError("Unsupported action space type.")

        return Batch(act=actions, state=state)


class AlternatingPolicy(RuleBasedPolicy):
    """Alternate between choosing the action with index 0 and the action with the highest index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def forward(self, batch: Batch, state=None, **kwargs):
        if isinstance(self.action_space, Discrete):
            actions = np.where(self.count % 2 == 0, 0, self.action_space.n - 1)
            actions = np.full((len(batch.obs),), actions, dtype=int)
        elif isinstance(self.action_space, Box):
            if self.count % 2 == 0:
                actions = np.tile(self.action_space.low, (len(batch.obs), 1))
            else:
                actions = np.tile(self.action_space.high, (len(batch.obs), 1))
        else:
            raise ValueError("Unsupported action space type.")

        self.count += 1
        return Batch(act=actions, state=state)

