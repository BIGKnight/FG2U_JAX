"""Datasets used in examples."""
import jax.numpy as jnp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.array(pic.permute(1, 2, 0), dtype=jnp.float32)
    
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

PATH = 'data/CIFAR/'

def mnist(permute_train=False):
    # --------- Data Loading ---------#
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            FlattenAndCast(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            FlattenAndCast(),
        ]
    )

    train_dataset = MNIST(
        root=PATH, train=True, download=False, transform=transform_train
    )

    test_dataset = MNIST(
        root=PATH, train=False, download=False, transform=transform_test
    )
    return train_dataset, test_dataset
