""" Main qimage module """

import os
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import TensorDataset, DataLoader

from img_transforms import *

def get_dataset(
        dataset_name="fashion-mnist",
        batch_size=128,
        transforms=None,
        transform_labels=None,
        overwrite=False):
    """ Returns a dataset with the applied transformations.
    Args:
        dataset_name: str, one of `mnist` or `fashion-mnist`
        batch_size: int
        transforms: A list of transforms to apply.
        transform_labels: List of strings, labels for each transform.
        overwrite: bool, if True, overwrites existing files
    Returns:
        train, test: Datasets of train and test
    """

    if transforms is None:
        transforms = []
        transform_labels = []

    assert len(transforms) == len(transform_labels)

    if dataset_name == "fashion-mnist":
        dataset_fn = FashionMNIST
    elif dataset_name == "mnist":
        dataset_fn = MNIST
    else:
        raise ValueError("Invalid dataset name. Choose `mnist` or `fashion-mnist`")

    composed_transforms = Compose(transforms)
    dirname = get_dirname(dataset_name, [tl for tl in transform_labels if tl is not None])

    if os.path.exists(dirname + "train_data.pt") and not overwrite:
        train = TensorDataset(torch.load(dirname + "train_data.pt"),
                    torch.load(dirname + "train_targets.pt"))
        test = TensorDataset(torch.load(dirname + "test_data.pt"),
                    torch.load(dirname + "test_targets.pt"))
    else:
        train, test = transform_and_cache(dataset_fn, composed_transforms, dirname)

    return train, test

def transform_and_cache(dataset_fn, transforms, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    train = dataset_fn('/tmp/vision/', download=True, transform=transforms, train=True)
    test = dataset_fn('/tmp/vision/', download=True, transform=transforms, train=False)
    train_dl = DataLoader(train, batch_size=len(train.data))
    test_dl = DataLoader(test, batch_size=len(test.data))

    train_data, train_targets = next(iter(train_dl))
    test_data, test_targets = next(iter(test_dl))

    torch.save(train_data, dirname + "train_data.pt")
    torch.save(test_data, dirname + "test_data.pt")
    torch.save(train_targets, dirname + "train_targets.pt")
    torch.save(test_targets, dirname + "test_targets.pt")

    return train, test

def get_dirname(dataset_name, transform_labels):
    directory = "/".join(transform_labels)
    return dataset_name + "/" + directory + "/"

if __name__ == '__main__':
    dataset_name="fashion-mnist"
    batch_size=128
    transforms=[Resize((32, 32)), ToTensor(), Channel("fourier"), Flatten()]
    transform_labels=["size_32x32", None, "kernel_fourier", None]
    get_dataset(dataset_name, batch_size, transforms, transform_labels, overwrite=True)

