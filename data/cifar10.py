# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""
import jax.numpy as jnp
import numpy as np
from torch.utils import data
from torchvision.datasets import CIFAR10
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

# def _one_hot(x, k, dtype=np.float32):
#     """Create a one-hot encoding of x of size k."""
#     return np.array(x[:, None] == np.arange(k), dtype)

def cifar10(permute_train=False):
    # --------- Data Loading ---------#
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomCrop(
            #     (32, 32),
            #     padding=4,
            #     fill=0,
            #     padding_mode="constant",
            # ),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize(
            #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            # ),
            FlattenAndCast(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            # ),
            FlattenAndCast(),
        ]
    )

    train_dataset = CIFAR10(
        root=PATH, train=True, download=False, transform=transform_train
    )

    test_dataset = CIFAR10(
        root=PATH, train=False, download=False, transform=transform_test
    )
    # train_images = train_dataset.data
    # train_labels = jnp.array(train_dataset.targets)
    # test_images = test_dataset.data
    # test_labels = jnp.array(test_dataset.targets)
    # train_labels = _one_hot(train_labels, 10)
    # test_labels = _one_hot(test_labels, 10)

    # if permute_train:
    #     perm = np.random.RandomState(0).permutation(train_images.shape[0])
    #     train_images = train_images[perm]
    #     train_labels = train_labels[perm]
        
    return train_dataset, test_dataset
    # return train_images, train_labels, test_images, test_labels

    # train_loader = NumpyLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=False,
    # )

    # test_loader = NumpyLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=False,
    # )
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)