""" Handles all data loading and preprocessing with PyTorch loaders. The main
contribution is the conversion of the images to a single tensor of batched
matrix product states. This file also implements the caching protocols
necessary to train at a reasonable pace, since converting a vector to an MPS
is in general an expensive operation. 
"""

from typing import Generator, Tuple, Mapping, Callable
import jax
import jax.numpy as jnp
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Resize, Compose, ToTensor

import numpy as np

dataset_fns = dict(mnist=MNIST, fashion_mnist=FashionMNIST)

def check_param_saturation(img_size, pd, chi_img):
    """ Checks whether the mps method actually does any compression """
    Npatches = np.prod(img_size) / np.prod(pd)
    Npx = 2 * np.prod(pd) # 2 channels
    L = int(np.ceil(np.log2(Npx)))
 
    chi_max = 2**(L//2)
    print("chi_max: ", int(chi_max))
    return chi_img <= chi_max

# Dataloaders / datasets
def numpy_collate(batch):
    """ Collate function for Numpy dataloader """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(DataLoader):
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
    """ Dataset in matrix product state form. """
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
        """
        Args:
            dataset_name: string, either mnist or fashion_mnist
            resize: tuple of size to scale images to
            chi_max: Bond dimension of images
            patch_dim: Size of patches. If patch_dim=resize, then only logN 
                qubits are used.
            fpath: Location to save processed images.
            kernel: String representing local Hilbert space function. One of
                'diff' (Google), 'rot' (Stoudenmire), 'fourier'.
            train: bool, train vs test set.
        """
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


def mps_norm(mps):
    L = len(mps)
    out = mps[0]
    lenv = np.tensordot(mps[0], mps[0].conj(), [0,0]).transpose((0,2,1,3))
    for i in range(1, L):
        tnsr = np.tensordot(mps[i], mps[i].conj(), [0,0]).transpose((0,2,1,3))
        lenv = np.tensordot(lenv, tnsr, [[2,3],[0,1]])
    return np.sqrt(lenv[0,0,0,0])

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
    def __init__(self, method):
        self.method = method
    def __call__(self, img):
        if self.method == "diff":
            return torch.cat((1-img, img), 0)
        elif self.method == "rot":
            return torch.cat((np.cos(0.5*np.pi*img), np.sin(0.5*np.pi*img)), 0)
        elif self.method == "fourier":
            f = torch.fft.fft2(img)
            f /= np.prod(img.shape[-2:])
            return torch.cat((torch.real(f), torch.imag(f)), 0)

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
            norm = mps_norm(mps)
            if norm != 0.:
                mps[0] /= norm
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
    return f"size_{resize[0]}x{resize[1]}_patch_{patch_dim[0]}x"\
                f"{patch_dim[1]}_chi{chi_max}_kernel_{kernel}"

def cache_transformed_dataset(
        dataset_name: str="mnist",
        *,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff"):
    needs_cleanup = _cache_transformed_dataset(
                dataset_name,
                resize=resize,
                fpath=fpath,
                chi_max=chi_max,
                patch_dim=patch_dim,
                kernel=kernel)
    if needs_cleanup:
        _collect_and_cleanup(
                dataset_name,
                resize=resize,
                fpath=fpath,
                chi_max=chi_max,
                patch_dim=patch_dim,
                kernel=kernel)

def _collect_and_cleanup(
        dataset_name: str="mnist",
        *,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff",
        Nbatches: int=10):
    testx, testy = [], []
    trainx, trainy = [], []

    fname = dataset_fname(resize=resize,
            chi_max=chi_max, 
            patch_dim=patch_dim,
            kernel=kernel)
    dirname = f"{fpath}/{dataset_name}/"


    for i in range(Nbatches):
        trainx.append(torch.load(f"{dirname}/train_data{i}_{fname}.pt"))
        trainy.append(torch.load(f"{dirname}/train_targets{i}_{fname}.pt"))
        testx.append(torch.load(f"{dirname}/test_data{i}_{fname}.pt"))
        testy.append(torch.load(f"{dirname}/test_targets{i}_{fname}.pt"))


    testx = np.concatenate(testx, axis=0)
    testy = np.concatenate(testy, axis=0)
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)

    # TODO  This only reaches here if those files exist...but even so we should
    # have more error checking

    torch.save(trainx, f"{dirname}/train_data_{fname}.pt")
    torch.save(trainy, f"{dirname}/train_targets_{fname}.pt")
    torch.save(testx, f"{dirname}/test_data_{fname}.pt")
    torch.save(testy, f"{dirname}/test_targets_{fname}.pt")


    for i in range(Nbatches):
        os.remove(f"{dirname}/train_data{i}_{fname}.pt")
        os.remove(f"{dirname}/train_targets{i}_{fname}.pt")
        os.remove(f"{dirname}/test_data{i}_{fname}.pt")
        os.remove(f"{dirname}/test_targets{i}_{fname}.pt")

def _cache_transformed_dataset(
        dataset_name: str="mnist",
        *,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff",
        Nbatches: int=10):
    if patch_dim is None:
        patch_dim = resize
    fname = dataset_fname(resize=resize,
            chi_max=chi_max, 
            patch_dim=patch_dim,
            kernel=kernel)
    dirname = f"{fpath}/{dataset_name}/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # check if ALL files exist
    # train_data, train_targets, test_data, test_targets
    prepends = ["train_data", "train_targets", "test_data", "test_targets"]
    already_cached = True
    for prepend in prepends:
        if not os.path.exists(f"{dirname}/{prepend}_{fname}.pt"):
            already_cached = False
    if already_cached:
        return False

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

    Ntrain, Ntest = len(train_dataset.data), len(test_dataset.data)

    train_dl = NumpyLoader(train_dataset, batch_size=Ntrain//Nbatches)
    test_dl = NumpyLoader(test_dataset, batch_size=Ntest//Nbatches)

    del train_dataset
    del test_dataset

    for i in range(Nbatches):
        print(f"Cached {i} of {Nbatches}")
        train_img, train_targets = next(iter(train_dl))
        test_img, test_targets = next(iter(test_dl))

        torch.save(train_img, f"{dirname}/train_data{i}_{fname}.pt")
        torch.save(train_targets, f"{dirname}/train_targets{i}_{fname}.pt")
        torch.save(test_img, f"{dirname}/test_data{i}_{fname}.pt")
        torch.save(test_targets, f"{dirname}/test_targets{i}_{fname}.pt")
    return True

def load_training_set(
        dataset_name: str="mnist",
        *,
        batch_size: int,
        resize: tuple=(32,32),
        fpath: str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff") -> NumpyLoader:
    dataset = MPSCompressed(
                dataset_name=dataset_name,
                resize=resize,
                chi_max=chi_max,
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
        resize: tuple=(32,32),
        fpath:str="processed_datasets/",
        chi_max: int=1,
        patch_dim: tuple=None,
        kernel: str="diff"
        ) -> NumpyLoader:
    datasets = []

    for train in [True, False]:
        datasets.append(
            MPSCompressed(
                dataset_name=dataset_name,
                resize=resize,
                chi_max=chi_max,
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
