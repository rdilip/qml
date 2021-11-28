import numpy as np
from .utils import to_mps, pad_to_umps, mps_norm
import torch
from torchvision.transforms import Resize, Compose, ToTensor


class ToNumpyArray(object):
    def __call__(self, img):
        return np.array(img)

class ToTensor(ToTensor):
    def __str__(self):
        return "torch_tensor"

class Resize(Resize):
    def __str__(self):
        size = self.size
        return f"resize_{size[0]}x{size[1]}"

class FlattenPatches(object):
    def __call__(self, img):
        Nc, Npy, Npx, H, W = img.shape
        return np.transpose(img, (1, 2, 3, 4, 0)).reshape((Npx*Npy, H*W*Nc))

    def __str__(self):
        return "flatten_patches"

class ToPatches(object):
    def __init__(self, pd):
        """ pd is a tuple of number of patches along each axis, e.g., (4, 2)
        means 4 patches along y axis and 2 patches along x """
        self.pd = pd

    def __call__(self, img):
        pd = self.pd
        C, H, W = img.shape
        assert (H % pd[0] == 0 and W % pd[1] == 0), "pd must be evenly divisible"\
            "by image size"
        Wp, Hp = W // pd[1], H // pd[0]
        patches = torch.Tensor(img).unfold(-2, Hp, Hp).unfold(-2, Wp, Wp)
        return patches

    def __str__(self):
        pd = self.pd
        return f"patches_{pd[0]}x{pd[1]}"


class NormalizeVector(object):
    def __call__(self, img):
        return img / np.linalg.norm(img)

    def __str__(self):
        return "normalize"


class NormalizePatches(object):
    def __call__(self, img):
        Npatches, Npixels = img.shape
        return img / np.linalg.norm(img, axis=-1)[:, None]

    def __str__(self):
        return "normalizes"


class NormalizeMPS(object):
    def __call__(self, mps):
        norm = mps_norm(mps)
        mps[0] /= norm
        return mps

    def __str__(self):
        return "normalize"


class RelativeNormMPS(object):
    """ This converts an image (possibly patched) to an MPS with an additional
    qubit encoding the norm. Will throw an error if no actual compression is
    being done.
    Args:
        chi_max: int, maximum bond dimension.
        img: np.array of shape (Npatches, Npix), where Npix is the length of
            the vector to be converted to an MPS.
    """

    def __init__(self, chi_max):
        self.chi_max = chi_max

    def __call__(self, img):
        chi = self.chi_max
        Npatches, Npix = img.shape

        norms = np.linalg.norm(img, axis=-1)

        if Npix == 1:
            return np.hstack((np.cos(0.5*np.pi*img), np.sin(0.5*np.pi*img))).reshape((-1, 2, 1, 1))

        norm_qubits = np.array(norms) / np.max(norms)
        norm_qubits = np.vstack((np.cos(0.5*np.pi*norm_qubits),
                                 np.sin(0.5*np.pi*norm_qubits))).T

        zero_norms = np.isclose(norms, 0.)
        Nzero = np.sum(zero_norms)
        img[zero_norms] = torch.ones(
            (np.sum(zero_norms), Npix)) / np.sqrt(Npix)

        norms[zero_norms] = 1.0
        img /= norms[:, None]
        img = ToMPS(chi, append=norm_qubits, normalize=False)(img)
        Npatches, L, d, chi, _ = img.shape
        return img.reshape((L*Npatches, d, chi, chi))

    def __str__(self):
        return f"relative_norm_mps_chi{self.chi_max}"

class Snake(object):
    """ Reorders the pixels of the last two axes in `snake` order; possibly
    important to preserve locality
    """
    def __call__(self, img):
        img = np.array(img, dtype=np.float64)
        img[1::2] = img[1::2, ::-1]
        return img
    def __str__(self):
        return "snake"

class ColorQubitMPS(object):
    """ This converts an iamge (possibly patched) to an MPS with an additional
    color qubit. Will throw an error if no actual compression is being done.
    Args:
        chi_max: int, max bond dimension
        img: np.array of shape (Npatches, Npix), where Npix is the length of the
            vector to be converted to an MPS
    """

    def __init__(self, chi_max):
        self.chi_max = chi_max

    def __call__(self, img):
        chi = self.chi_max
        Npatches, Npix = img.shape
        if Npix == 1:
            print("Npix is 1")
            return np.hstack((np.cos(0.5*np.pi*img), np.sin(0.5*np.pi*img))).reshape((-1, 2, 1, 1))
        img = np.dstack((np.cos(0.5*np.pi*img), np.sin(0.5*np.pi*img)))
        img = img.reshape((Npatches, Npix*2))
        img = ToMPS(chi, append=None, normalize=True)(img)
        Npatches, L, d, chi, _ = img.shape
        return img.reshape((L*Npatches, d, chi, chi))

    def __str__(self):
        return f"color_qubit_mps_chi{self.chi_max}"

class ToMPS(object):
    def __init__(self, chi_max, append=None, normalize=False):
        self.chi=chi_max
        self.append=append
        self.normalize = normalize

    def __call__(self, batched_vector):
        # TODO: normalization
        chi, append = self.chi, self.append

        Npatches, N = batched_vector.shape
        L=int(np.ceil(np.log2(N)))  # Assumes channels for normalization
        if append is not None:
            L += 1
        batched_mps=np.zeros((Npatches, L, 2, chi, chi), dtype=np.float64)

        chi_sat=2**(L//2)
        if self.chi > chi_sat:
            raise ValueError(
                f"chi is greater than bond dimension saturation {chi_sat}")

        for a in range(Npatches):
            mps = to_mps(batched_vector[a], chi_max=self.chi)
            if append is not None:
                mps.append(append[a].reshape((2, 1, 1)))
            batched_mps[a] = pad_to_umps(mps)
        batched_mps = np.array(batched_mps, dtype=np.float64).reshape((Npatches, L, 2, chi, chi))
        if self.normalize:
            norms = np.array([mps_norm(mps) for mps in batched_mps], dtype=np.float64)
            assert not np.any(np.isclose(norms, 0.))
            batched_mps[:, 0] /= norms[:, np.newaxis, np.newaxis, np.newaxis]
        return batched_mps

    def __str__(self):
        return f"mps_chi{self.chi}"

class ToTrivialMPS(object):
    def __call__(self, vector):
        # always 2 channels, c comes first
        return vector.reshape((2, -1)).T.reshape((-1, 2, 1, 1))
    def __str__(self):
        return f"trivial_mps"
