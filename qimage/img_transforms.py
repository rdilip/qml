import numpy as np
from utils import *
import torch
from torchvision.transforms import Resize, Compose, ToTensor

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


