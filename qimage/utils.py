""" Utility functions for MPS """
import numpy as np
import jax.numpy as jnp
import jax

def check_param_saturation(vector_size, chi_img):
    """ Checks whether the provided compression is actually compressing the
    vector """
    L = int(np.ceil(np.log2(vector_size)))
    Nparams_mps = chi_img * chi_img * L * 2
    return vector_size >= Nparams_mps

def mps_norm(mps):
    return mps_overlap(mps, mps)

def mps_overlap(mps1, mps2):
    L = len(mps1)
    assert len(mps2) == L
    lenv = np.tensordot(mps1[0], mps2[0].conj(), [0,0]).transpose((0,2,1,3))
    for i in range(1, L):
        tnsr = np.tensordot(mps1[i], mps2[i].conj(), [0,0]).transpose((0,2,1,3))
        lenv = np.tensordot(lenv, tnsr, [[2,3],[0,1]])
    return np.sqrt(lenv[0,0,0,0])

def to_mps(invector, chi_max=10):
    N = invector.shape[-1]
    L = int(np.ceil(np.log2(N)))
    right = np.pad(invector, (0, 2**L-N))

    chiL = 1
    mps = []
    for i in range(L-1):
        left, s, B = np.linalg.svd(right.reshape(chiL*2, 2**(L-i-1)), full_matrices=False)
        left, s, B = left[:,:chi_max], s[:chi_max], B[:chi_max, :]
        mps.append(left.reshape((chiL, 2, -1)).transpose((1,0,2)))
        chiL = left.shape[-1]
        right = (np.diag(s) @ B).reshape((chiL, -1))
    mps.append(right.T.reshape((2,-1, 1)))
    return mps

def normalize_mps(mps):
    ovlp = mps_norm(mps)
    mps[0] /= ovlp
    return mps

def to_vector(mps):
    outvector = mps[0][:, 0, :]
    for i in range(1, len(mps)):
        outvector = np.tensordot(outvector, mps[i], [-1,-2])
    return outvector[..., 0].ravel()

def pad_to_umps(mps):
    L = len(mps)
    chi = jnp.max(jnp.array([i.shape[1:] for i in mps]))
    umps = []
    for i in range(L):
        _, chiL, chiR = mps[i].shape
        umps.append(jnp.pad(mps[i], ((0,0),(0,chi-chiL),(0,chi-chiR))))
    return jnp.array(umps)

def umps_to_vector(umps):
    mps = [np.array(a) for a in umps]
    mps[0] = mps[0][:,0,:].reshape((2,1,-1))
    mps[-1] = mps[-1][:,:,0].reshape((2,-1,1))
    return to_vector(mps)
