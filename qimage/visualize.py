""" Visualization functions """
import numpy as np
import matplotlib.pyplot as plt
from .utils import to_vector

def visualize_img(img, kernel=None, mps=False, **params):
    img = np.array(img)
    if mps is True:
        assert 'Nx' in params and 'Ny' in params
        Nx, Ny = params['Nx'], params['Ny']
        img = to_vector(img).reshape((Ny, Nx, 2)).transpose((2,0,1))
        visualize_img(img, kernel=kernel)

    Ny, Nx = img.shape[-2], img.shape[-1]
        
    if kernel is None:
        plt.imshow(img.reshape((Ny, Nx)))
    elif kernel == "diff":
        visualize_img(img[..., 1, :, :], kernel=None)


