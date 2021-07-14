""" Handles all data loading and preprocessing with PyTorch loaders.
NOTE: need to rewrite the structure of io since we're using pytorch, not
tf rn, but let's keep it backwards compatible.

"""

from typing import Generator, Tuple, Mapping, Callable
import jax
import jax.numpy as jnp
import numpy as np

import torch
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Resize, Compose, ToTensor

Batch = Mapping[str, np.ndarray]
import numpy as np

dataset_fns = dict(mnist=MNIST, fashion_mnist=FashionMNIST)

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class Flatten(object):
    def __call__(self, img):
        return np.reshape(img, (*img.shape[:-2], np.prod(img.shape[-2:])))

class SplitPatches(object):
    def __init__(self, ds):
        self.ds = ds
    def __call__(self, img):
        Nx, Ny = self.ds
        C, H, W = img.shape
        if H % Nx != 0 or W % Ny != 0:
            raise ValueError("Invalid number of patches, must divide evenly")
        px, py = H // Nx, W // Ny
        img_split = torch.Tensor(img).unfold(1,px,px).unfold(2,py,py)
        return np.array(img_split.reshape(C, -1, px, py), dtype=float).transpose((1,0,2,3))

class SplitAndFlatten(object):
    def __init__(self, dim_split):
        self.ds = dim_split

class Cast(object):
    def __call__(self, img):
        return torch.Tensor(img)
        # return jnp.array(img, dtype=jnp.float32)

class Channel(object):
    # TODO: Fourier, x, 1-x, etc.
    # TODO: WE NEED THE CHANNELS: IT MAKES NO SENSE IN THE LIMIT WITHOUT IT
    def __call__(self, img):
        return torch.cat((1-img, img), 0)

class ToMPS(object):
    def __init__(self, chi_max):
        self.chi = chi_max

    def pad_fn(self, A):
        d, chiL, chiR = A.shape
        return np.pad(A, ((0,0), (0,self.chi-chiL), (0,self.chi-chiR)))

    def __call__(self, batched_vector):
        # TODO make this much speed
        alpha, Nc, Npx = batched_vector.shape
        batched_vector = np.reshape(batched_vector, (alpha, Nc*Npx))
        L = int(np.ceil(np.log2(Npx*Nc)))
        dim = 2**L
        batched_vector = np.pad(batched_vector, ((0,0),(0,dim-Npx*Nc)))
        batched_vector = batched_vector.reshape((alpha,*[2]*L))
        batched_mps = np.zeros((alpha, L, 2, self.chi, self.chi))

        for a in range(alpha):
            vector = batched_vector[a]
            mps = []
            chiL = 1
            for i in range(L-1):
                vector = vector.reshape((chiL*2, 2**(L-i-1)))
                A, s, B = np.linalg.svd(vector, full_matrices=False)
                A, s, B = A[:,:self.chi], s[:self.chi], B[:self.chi,:]
                mps.append(self.pad_fn(A.reshape((chiL, 2, -1)).transpose((1,0,2))))
                vector = (np.diag(s)@B).reshape((-1, 2**(L-i-1)))
                chiL = vector.shape[0]
            mps.append(self.pad_fn(vector.reshape((chiL,2,1)).transpose((1,0,2))))
            batched_mps[a] = mps
        return np.array(batched_mps)


def load_training_set(
        dataset_name: str="mnist",
        *,
        batch_size: int,
        resize: tuple,
        chi_max: int,
        patch_dim: tuple
        ) -> NumpyLoader:
    DatasetFn = dataset_fns[dataset_name]
    dataset = DatasetFn(f'/tmp/{dataset_name}/',
                        download=True,
                        transform=Compose([Resize(resize),
                                           ToTensor(),
                                           Channel(), 
                                           SplitPatches(patch_dim),
                                           Flatten(),
                                           ToMPS(chi_max)
                                           ]))
    data_generator = NumpyLoader(dataset, 
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0)
    return data_generator

def load_eval_set(
        dataset_name: str="mnist",
        *,
        batch_size: int,
        resize: tuple,
        chi_max: int,
        patch_dim: tuple,
        ) -> NumpyLoader:
    DatasetFn = dataset_fns[dataset_name]
    transform = Compose([Resize(resize),
                   ToTensor(),
                   Channel(),
                   SplitPatches(patch_dim),
                   Flatten(),
                   ToMPS(chi_max)
                   ])

    dataset = DatasetFn(f'/tmp/{dataset_name}/',
                        download=True,
                        transform=transform,
                        train=True)
    train_eval = NumpyLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataset = DatasetFn(f'/tmp/{dataset_name}/',
                        download=True,
                        transform=transform,
                        train=False)
    test_eval = NumpyLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_eval, test_eval



