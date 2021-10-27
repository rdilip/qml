import numpy as np
from .utils import to_mps, pad_to_umps
import torch
from torchvision.transforms import Resize, Compose, ToTensor

class ToNumpyArray(object):
    def __call__(self, img):
        return np.array(img)

class Flatten(object):
    def __call__(self, img):
        Nc, Nx, Ny = img.shape
        img = np.reshape(np.array(img), (Nc, Nx*Ny))
        return img

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

    def __call__(self, batched_vector):
        # TODO: normalization
        alpha, N = batched_vector.shape  # alpha is number of patches, channels should be in N
        L = int(np.ceil(np.log2(N))) # Assumes channels for normalization
        batched_mps = np.zeros((alpha, L, 2, self.chi, self.chi))
        for a in range(alpha):
            mps = to_mps(batched_vector[a], chi_max=self.chi)
            batched_mps[a] = pad_to_umps(mps)
        return np.array(batched_mps)

class ToTrivialMPS(object):
    def __call__(self, vector):
        return vector.reshape((2, -1)).T.reshape((-1, 2, 1, 1)) # always 2 channels, c comes first

