""" Visualization functions. We do a lot of image processing here, so some of
these visualizations are quite expensive because they need to undo a lot of 
transformations. """
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import to_vector

def visualize_img(img,
                    patched=False,
                    scale=1,
                    size=(32,32),
                    **params):
    """ Visualizes img, where img is  of shape (*, Nc, Ny, Nx). """
    img = np.array(img)
    Ny, Nx = size

    if patched is True:
        assert "pd" in params
        pd = params["pd"]
    else:
        pd = (1,1)

    pNy, pNx = Ny // pd[0], Nx // pd[1]

    # Flattened image: unroll to (patch y, patch x, Npatch y, Npatch x)
    if len(img.shape) == 2:
        img = img.reshape((*pd, Ny//pd[0], Nx//pd[1]))

    if patched is True:
        fig = plt.figure(figsize=(4,4))
        
        for yp in range(pd[0]):
            for xp in range(pd[1]):
                ax = fig.add_subplot(pd[0], pd[1], yp*pd[1]+xp+1, xticks=[], yticks=[])
                visualize_img(img[..., yp, xp, :, :], patched=False, scale=scale, size=(pNy, pNx))
        plt.subplots_adjust(wspace=0, hspace=0)
        return 

    plt.imshow(img.reshape((Ny, Nx)), vmin=0, vmax=scale)

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


