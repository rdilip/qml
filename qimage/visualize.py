""" Visualization functions. We do a lot of image processing here, so some of
these visualizations are quite expensive because they need to undo a lot of 
transformations. """
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import to_vector

def visualize_patched_img(img, pd, size=(32,32), kernel=True, **kwargs):
    """ Visualizes a patched image where the patches are explicit. Note the 
    difference between this and explicitly unrolling the patches into one long
    vector. 
    Args:
        img: np.ndarray of shape (pd[0], pd[1], Ny*Nx*2)
        pd: tuple of shape (pdy, pdx), describing the splitting of the image.
        size: Original shape of image
    Returns:
        None
    """

    assert size[0] % pd[0] == 0 and size[1] % pd[1] == 0
    pNy, pNx = size[0] // pd[0], size[1] // pd[1]
    if kernel:
        img = img.reshape((*pd, pNy*pNx*2))
    else:
        img = img.reshape((*pd, pNy*pNx))

    fig = plt.figure(figsize=(4,4))
    
    for yp in range(pd[0]):
        for xp in range(pd[1]):
            ax = fig.add_subplot(pd[0], pd[1], yp*pd[1]+xp+1, xticks=[], yticks=[])
            visualize_vector(img[..., yp, xp, :], size=(pNy, pNx), kernel=kernel, **kwargs)
    plt.subplots_adjust(wspace=0, hspace=0)

def visualize_vector(vec, size=(32,32), kernel=True, normalized=True, scale=1):
    """ Displays a nonpatched vector. 
    Args:
        vec: Vector to be displayed
        size: tuple (Ny, Nx) s.t. Ny * Nx == len(vec)
        Kernel: bool, if true, assumes the vector has the form 
            np.dstack([np.cos(pi/2 * vec), np.sin(pi/2 * vec)]).ravel()
        scale: float, maximum value of colorplot
    Returns:
        None
    """
    Ny, Nx = size
    assert vec.shape[0] == Ny * Nx * (2 if kernel else 1)
    vec = np.array(vec, dtype=np.float64)
    if normalized:
        vec *= np.sqrt(np.prod(size))
    if kernel:
        vec = np.arccos(vec[::2]) * 2 / np.pi
    plt.imshow(vec.reshape((Ny, Nx)), vmin=0, vmax=scale)
    return vec

def mps_to_vector_img(img, norm_qubit=False):
    """ First index should be patches -- always! """
    if norm_qubit:
        norms = img[:, -1, 0, 0, 0]
        norms = (2/np.pi) * np.arccos(norms)
        img = np.array([to_vector(patch) for patch in img[:, :-1, ...]])
        img *= norms[:, None]
    else:
        img = np.array([to_vector(patch) for patch in img])
    return img


