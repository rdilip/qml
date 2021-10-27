""" A couple of visualizations because these mps things get complicated """

import matplotlib.pyplot as plt
import torch
import numpy as np 
from compress import to_vector 

def visualize(patches, colorbar=True):
    Nx, Ny = patches.shape[:2]
    fig = plt.figure(figsize=(4,4))
    axes = []
    for i in range(Nx):
        for j in range(Ny):
            inp = patches[i,j]
            ax = fig.add_subplot(Nx,Ny,i*Ny+j+1, xticks=[], yticks=[])
            im = ax.imshow(inp, vmin=torch.min(patches), vmax=torch.max(patches))
    if colorbar:
        cbaxes = fig.add_axes([1.0, 0.1, 0.03, 0.8]) 
        cbar = fig.colorbar(im, cax = cbaxes)
    plt.subplots_adjust(wspace=0, hspace=0.05)


def visualize_mps(batched_img_mps, img_shape, patch_dim, **kwargs):
    Ny, Nx = int(img_shape[0]/patch_dim[0]), int(img_shape[1]/patch_dim[1])
    batched_img = np.array([to_vector(mps) for mps in batched_img_mps])
    batched_img = batched_img.reshape((*patch_dim, Ny, Nx))
    visualize(torch.Tensor(batched_img), **kwargs)
