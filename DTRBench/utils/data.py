import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data import Batch
from typing import List, Optional, Callable, Dict, Any, Union, Tuple
import tianshou
from torch.utils.data import Subset
import h5py


def load_buffer(buffer_path: str) -> ReplayBuffer:
    buf = ReplayBuffer.load_hdf5(buffer_path)
    buf._size = len(buf._meta)
    return buf


class TianshouDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer, stack_num=1, **kwargs):
        self.buffer = buffer
        self.stack_num = stack_num
        self.buffer.stack_num = self.stack_num
        self.episode_start_indices = self._get_episode_start_indices()
        self.check_buffer()

    def _get(self, index) -> Batch:
        current_episode_start = self.episode_start_indices[self.episode_start_indices <= index][-1]

        # Calculate the real length of the episode
        real_length = index - current_episode_start + 1

        # Generate the mask
        mask = np.zeros(self.stack_num, dtype=bool)
        mask[-real_length:] = True  # tianshou uses pre-padding, so the last real_length elements are valid

        # Append the mask to the Batch
        data = self.buffer[index]
        data.mask = mask
        data.to_numpy()
        return data

    def __getitem__(self, index):
        batch = self._get(index)
        return batch

    def __len__(self):
        return len(self.buffer._meta)

    def check_buffer(self):
        done_flags = self.buffer.done

        episode_starts = self._get_episode_start_indices()

        # Check if last step of each episode has 'done' as True
        episode_ends = episode_starts[1:] - 1  # indices of episode ends
        episode_ends = np.append(episode_ends, len(done_flags) - 1)  # include the end of the last episode
        if not all(done_flags[end] == True for end in episode_ends):
            raise ValueError("The buffer must only contain complete episodes! ")

        # check if terminated and truncated are mutually exclusive at the end of each episode
        if not all(np.logical_xor(self.buffer.truncated[episode_ends], self.buffer.terminated[episode_ends])):
            raise ValueError("truncate and terminate are not mutually exclusive at the end of each episode! ")

        # check if the intermediate steps have truncated or terminated as False
        if not all(np.logical_or(self.buffer.truncated[episode_starts[1:] - 1],
                                 self.buffer.terminated[episode_starts[1:] - 1])):
            raise ValueError("Intermediate steps have truncated or terminated as True! ")

        # Check if the last step of the buffer has 'done' as True
        if not done_flags[-1]:
            raise ValueError("The buffer must only contain complete episodes! ")

        return True

    def _get_episode_start_indices(self):
        done_indices = np.where(self.buffer.done)[0]  # find 'done' indices
        # Shift indices by 1 to get start of next episode, and prepend 0 for the first episode
        episode_start_indices = np.roll(done_indices + 1, shift=1)
        episode_start_indices[0] = 0  # set first index to 0
        return episode_start_indices


def collate_batch_seq2seq(batch: List[tianshou.data.Batch],
                          device: torch.device = torch.device("cpu"),
                          dtype: np.dtype = torch.float32) -> tianshou.data.Batch:
    b = Batch.stack(batch)
    b.to_torch(device=device, dtype=dtype)
    return b


def collate_batch_seq2one(batch: List[tianshou.data.Batch],
                          device: torch.device = torch.device("cpu"),
                          dtype: np.dtype = torch.float32) -> tianshou.data.Batch:
    b = Batch.stack(batch)
    b.to_torch(device=device, dtype=dtype)
    b.y = b.y[:, -1, :]
    b.mask = b.mask[:, -1, :]
    return b


class TianshouEpisodeDataset(TianshouDataset):
    """
    Caution! This dataset is ONLY suitable for buffer collecting from a SINGLE env.
    """

    def __init__(self, buffer, pad="pre", **kwargs):
        super().__init__(buffer, stack_num=1)
        self.pad = pad
        if self.pad not in ["pre", "post"]:
            raise NotImplementedError("Only pad=pre or post is supported for batch collation.")
        # Find the indices where 'done' is True
        # get max episode length
        self.max_len = np.diff(self.episode_start_indices, append=len(self.buffer._meta)).max()

    def _get(self, index):
        start = self.episode_start_indices[index]
        if index + 1 < len(self.episode_start_indices):
            end = self.episode_start_indices[index + 1]
        else:
            end = len(self.buffer._meta)
        assert end > start
        index_list = list(range(start, end))
        mask = [1] * len(index_list)

        rem_len = self.max_len - end + start
        # Pad each field in episode_data to self.max_len.
        if self.pad == "pre":  # pre-padding is preferred for RNNs due to the forgetting problem
            index_list = [start] * rem_len + index_list
            mask = [0] * rem_len + mask
        elif self.pad == "post":
            index_list = index_list + [end - 1] * rem_len
            mask = mask + [0] * rem_len
        assert len(mask) == self.max_len
        data = self.buffer[index_list]
        data["mask"] = np.expand_dims(mask, axis=-1)
        data.to_numpy()
        return data

    def __len__(self):
        return len(self.episode_start_indices)

    def __getitem__(self, index):
        batch = self._get(index)
        return batch


def episode_random_split(dataset: Union[TianshouDataset, TianshouEpisodeDataset],
                         ratios: List[float]) -> List[Subset]:
    assert sum(ratios) == 1.0, "The sum of the ratios should be 1.0"

    # Get the start indices of each episode
    episode_start_indices = dataset._get_episode_start_indices()

    num_episodes = len(episode_start_indices)

    # Compute the number of episodes for each split
    sizes = [int(num_episodes * ratio) for ratio in ratios]
    # If there are any remaining episodes due to rounding, add them to the last split
    sizes[-1] += num_episodes - sum(sizes)

    # Randomly permute the episode indices
    permuted_indices = torch.randperm(num_episodes)

    # Convert episode indices to sample indices and create a Subset for each split
    datasets = []
    start = 0
    for size in sizes:
        end = start + size
        indices = permuted_indices[start:end]
        if issubclass(dataset.__class__, TianshouDataset) and not issubclass(dataset.__class__, TianshouEpisodeDataset):
            sample_indices = [idx for episode_idx in indices for idx in range(episode_start_indices[episode_idx],
                                                                              episode_start_indices[
                                                                                  episode_idx + 1] if episode_idx + 1 < num_episodes else len(
                                                                                  dataset))]
        elif issubclass(dataset.__class__, TianshouDataset) and issubclass(dataset.__class__, TianshouEpisodeDataset):
            sample_indices = indices.tolist()
        else:
            raise NotImplementedError
        datasets.append(Subset(dataset, sample_indices))
        start = end
    return datasets


class ImitationEpisodeDataset(TianshouEpisodeDataset):
    def __getitem__(self, index):
        batch = self._get(index)
        x = torch.from_numpy(batch.obs)
        y = torch.from_numpy(batch.act)
        mask = torch.from_numpy(batch.mask)
        return Batch(x=x, y=y, mask=mask)


class ImitationDataset(TianshouDataset):
    def __getitem__(self, index):
        batch = self._get(index)
        x = torch.from_numpy(batch.obs)
        y = torch.from_numpy(batch.act)
        mask = torch.from_numpy(batch.mask)
        return Batch(x=x, y=y, mask=mask)


class SSLDataset(TianshouEpisodeDataset):
    def __getitem__(self, index):
        batch = self._get(index)
        x = torch.from_numpy(batch.obs)
        y = torch.from_numpy(batch.obs_next)
        mask = torch.from_numpy(batch.mask)
        return x, y, mask


class SeqPredLOSDataset(TianshouEpisodeDataset):
    num_cls = 4

    def __getitem__(self, index):
        batch = self._get(index)
        x = torch.from_numpy(batch.obs)
        LOS = torch.from_numpy(batch.info["LOS"])
        mask = torch.from_numpy(batch.mask)

        LOS = self.bin(LOS, 0, 40, self.num_cls)
        return x, LOS, mask

    def bin(self, LOS: torch.Tensor, min_val, max_val, num_classes):
        if num_classes <= 0 or num_classes > (max_val - min_val):
            raise ValueError(
                f"The number of bins must be greater than 0 and less than or equal to the range of the data ({max_val - min_val}).")
        bin_size = (max_val - min_val) / num_classes
        for i in range(num_classes):
            lower_bound = min_val + i * bin_size
            upper_bound = lower_bound + bin_size
            LOS[torch.logical_and(lower_bound <= LOS, LOS < upper_bound)] = i
        LOS[LOS >= upper_bound] = num_classes - 1
        return LOS.long()


class PredLOSDataset(TianshouDataset):
    num_cls = 4

    def __getitem__(self, index):
        batch = self._get(index)
        x = torch.from_numpy(batch.obs)
        LOS = torch.from_numpy(batch.info["LOS"]).to(torch.long)

        # if num_stack > 1, take the last LOS as ground truth
        if len(LOS.shape) == 2:
            LOS = LOS[-1, :]
        elif len(LOS.shape) == 3:
            raise NotImplementedError

        LOS = self.bin(LOS, 0, 40, self.num_cls)
        return x, LOS, torch.ones(*LOS.shape)  # in non-sequential case, mask is all 1

    def bin(self, LOS, min_val, max_val, num_classes):
        if num_classes <= 0 or num_classes > (max_val - min_val):
            raise ValueError(
                f"The number of bins must be greater than 0 and less than or equal to the range of the data ({max_val - min_val}).")
        bin_size = (max_val - min_val) / num_classes
        for i in range(num_classes):
            lower_bound = min_val + i * bin_size
            upper_bound = lower_bound + bin_size
            LOS[torch.logical_and(lower_bound <= LOS, LOS < upper_bound)] = i
        LOS[LOS >= upper_bound] = num_classes - 1
        return LOS


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    buffer_path = "../DTRGym/OberstSepsisEnv/offline_data/OberstSepsisEnv_imbalance_buffer_10000_ep0.05_rep0.25.hdf5"
    buffer = load_buffer(buffer_path)
    dataset = TianshouDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch_seq2seq)
    for b in dataloader:
        print(b.obs.shape)
        break

    dataset2 = TianshouEpisodeDataset(buffer)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True, collate_fn=collate_batch_seq2seq)
    for b in dataloader2:
        print(b.obs.shape)
