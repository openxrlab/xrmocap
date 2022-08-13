import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Union


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    """

    def __init__(self,
                 dataset,
                 n_replicas: Union[None, int] = None,
                 rank: Union[None, int] = None,
                 local_rank: Union[None, int] = None,
                 local_size: Union[None, int] = None,
                 shuffle: bool = True):
        """
        Args:
            dataset:
                Dataset used for sampling.
            n_replicas (Union[None, int], optional):
                Number of processes participating in distributed
                training. Defaults to None.
            rank (Union[None, int], optional):
                Rank of the current process within n_replicas.
                Defaults to None.
            local_rank (Union[None, int], optional):
                Defaults to None.
            local_size (Union[None, int], optional):
                Defaults to None.
            shuffle (bool, optional):
                Whether to shuffle the data. Defaults to True.

        """
        if n_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            n_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.n_replicas = n_replicas
        self.rank = rank
        self.epoch = 0
        self.n_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.n_replicas))
        self.total_size = self.n_samples * self.n_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indexes = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indexes = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indexes += indexes[:(self.total_size - len(indexes))]
        assert len(indexes) == self.total_size

        # subsample
        offset = self.n_samples * self.rank
        indexes = indexes[offset:offset + self.n_samples]
        assert len(indexes) == self.n_samples

        return iter(indexes)

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
