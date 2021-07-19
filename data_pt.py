""" Handles all data loading and preprocessing with PyTorch loaders.
NOTE: need to rewrite the structure of io since we're using pytorch, not
tf rn, but let's keep it backwards compatible.

"""

from typing import Generator, Tuple, Mapping, Callable
import jax
import jax.numpy as jnp
import numpy as np
import os

import torch
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Resize, Compose, ToTensor

Batch = Mapping[str, np.ndarray]
import numpy as np

dataset_fns = dict(mnist=MNIST, fashion_mnist=FashionMNIST)

# Dataloaders / datasets
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

class MPSCompressed(Dataset):
    def __init__(
            self,
            *,
            dataset_name,
            resize,
            chi_max,
            patch_dim,
            fpath,
            kernel,
            train=True
            ):
        fname = dataset_fname(
                resize=resize,
                chi_max=chi_max, 
                patch_dim=patch_dim,
                kernel=kernel)
        
        prepend = "train" if train else "test"
        self.data = torch.load(f"{fpath}/{dataset_name}/{prepend}_data_{fname}.pt")
        self.targets = torch.load(f"{fpath}/{dataset_name}/{prepend}_targets_{fname}.pt")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

# Transforms 

class Flatten(object):
    def __call__(self, img):
        img = np.reshape(img, (*img.shape[:-2], np.prod(img.shape[-2:])))
        return np.array(img).T

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

class Cast(object):
    def __call__(self, img):
        return torch.Tensor(img)
        # return jnp.array(img, dtype=jnp.float32)

class Channel(object):
    def __call__(self, img, method="diff"):
        if method == "diff":
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
        return np.array(batched_mps).reshape((alpha*L, 2, self.chi, self.chi))

class ToTrivialMPS(object):
    def __call__(self, vector):
        Npx, Nc = vector.shape
        return vector.reshape((Npx, Nc, 1, 1))

# Caching 
def dataset_fname(
        *,
        resize: tuple=(32,32), 
        chi_max: int=1, 
        patch_dim:tuple=None,
        kernel: str="diff"):
    if patch_dim is None:
        patch_dim = resize
    return f"size{resize[0]}x{resize[1]}_pd{patch_dim[0]}x"\
                "{patch_dim[1]}_chi{chi_max}_kernel_{kernel}.pt"

def cache_transformed_dataset(
        dataset_name: str="mnist",
        *,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff"):
    if patch_dim is None:
        patch_dim = resize
    fname = dataset_fname(resize, chi_max, patch_dim, kernel)
    dirname = f"{fpath}/{dataset_name}/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    transforms = Compose([
        Resize(resize),
        ToTensor(),
        Channel(method=kernel),
        SplitPatches(patch_dim),
        Flatten(),
        ToMPS(chi_max)
        ])

    train_dataset = dataset_fns[dataset_name](
            '/tmp/mnist/', download=True, transform=transforms, train=True)
    test_dataset = dataset_fns[dataset_name](
            '/tmp/mnist/', download=True, transform=transforms, train=False)

    train_dl = NumpyLoader(train_dataset, batch_size=len(train_dataset.data))
    test_dl = NumpyLoader(train_dataset, batch_size=len(train_dataset.data))

    train_img, train_targets = next(iter(train_dl))
    test_img, test_targets = next(iter(test_dl))

    torch.save(train_img, f"{dirname}/train_data_{fname}.pt")
    torch.save(train_targets, f"{dirname}/train_targets_{fname}.pt")
    torch.save(test_img, f"{dirname}/test_data_{fname}.pt")
    torch.save(test_targets, f"{dirname}/test_targets_{fname}.pt")

def load_training_set(
        dataset_name: str="mnist",
        *,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff"):
        ) -> NumpyLoader:
            self,
            *,
            dataset_name,
            resize,
            chi_max,
            patch_dim,
            fpath,
            train=True
    dataset = MPSCompressed(
                dataset_name=dataset_name,
                resize=resize,
                chi_max=chi_max
                patch_dim=patch_dim,
                fpath=fpath,
                kernel=kernel,
                train=True)

    data_generator = NumpyLoader(dataset, 
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1)
    return data_generator

def load_eval_set(
        dataset_name: str="mnist",
        *,
        batch_size: int,
        resize: tuple,
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff"
        ) -> NumpyLoader:
    for train in [True, False]:
    datasets.append(
            MPSCompressed(
                dataset_name=dataset_name,
                resize=resize,
                chi_max=chi_max
                patch_dim=patch_dim,
                fpath=fpath,
                kernel=kernel,
                train=train)
            )

    data_generators = [NumpyLoader(dataset, 
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=1) for dataset in datasets]
    return tuple(data_generators)
