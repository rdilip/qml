""" Short script to quickly run a model without doing any additional learning. """
import jax
import jax.numpy as jnp
from itertools import cycle
from torch.utils.data import DataLoader

from tn_mps import *

from qimage import qimage
from qimage.img_transforms import Resize, ToMPS, ToTrivialMPS, NormalizeVector,\
    ToPatches, NormalizeMPS, FlattenPatches, NormalizePatches, RelativeNormMPS, ColorQubitMPS, Snake

def load_model(shape, pd, chi_img, chi_tn, **dataset_params):
    dataset_name = dataset_params['dataset_name']
    model = np.load(f"output/{dataset_name}/size_{shape[0]}x{shape[1]}/" +\
                f"patch_{pd[0]}x{pd[1]}/chi_img{chi_img}/chi{chi_tn}_model.npy",\
                allow_pickle=True)
    return model[-1]

def run_model(shape, pd, chi_img, chi_tn, **dataset_params):
    tn = load_model(shape, pd, chi_img, chi_tn, **dataset_params)
    train, test = qimage.get_dataset(**dataset_params)
    train_eval = DataLoader(train, batch_size=len(test), collate_fn=qimage.numpy_collate)
    test_eval = DataLoader(test, batch_size=len(test), collate_fn=qimage.numpy_collate)
    train_eval, test_eval = cycle(train_eval), cycle(test_eval)
    breakpoint()
    print("Test accuracy: " + accuracy(tn, next(test_eval)))
    #print("Train accuracy: " + accuracy(tn, next(train_eval)))

if __name__ == '__main__':
    pd = (4,4)
    shape = (32,32)
    chi_img = 4
    chi_tn = 16

    dataset_params = dict(transforms=[Resize(shape), ToPatches(pd),\
            FlattenPatches(), Snake(), ColorQubitMPS(chi_img)],\
            dataset_name="fashion-mnist")

    run_model(shape, pd, chi_img, chi_tn, **dataset_params)
