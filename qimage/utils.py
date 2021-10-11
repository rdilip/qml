""" Utility functions for MPS """
import numpy as np
import jax.numpy as jnp

def check_param_saturation(img_size, pd, chi_img):
    """ Checks whether the mps method actually does any compression """
    Npatches = np.prod(img_size) / np.prod(pd)
    Npx = 2 * np.prod(pd) # 2 channels
    L = int(np.ceil(np.log2(Npx)))
 
    chi_max = 2**(L//2)
    print("chi_max: ", int(chi_max))
    return chi_img <= chi_max

def mps_norm(mps):
    L = len(mps)
    out = mps[0]
    lenv = np.tensordot(mps[0], mps[0].conj(), [0,0]).transpose((0,2,1,3))
    for i in range(1, L):
        tnsr = np.tensordot(mps[i], mps[i].conj(), [0,0]).transpose((0,2,1,3))
        lenv = np.tensordot(lenv, tnsr, [[2,3],[0,1]])
    return np.sqrt(lenv[0,0,0,0])

def to_mps(vector, chi_max=30, normalize=False):
    L = int(np.ceil(np.log2(len(vector))))
    dim = 2**L
    vector_padded = np.pad(vector, (0,dim-len(vector)))
    
    vector = vector_padded.reshape([2]*L)
    mps = []
    chiL = 1
    
    for i in range(L):
        vector = vector.reshape((chiL*2, 2**(L-i-1)))
        A, s, B = np.linalg.svd(vector, full_matrices=False)
        A, s, B = A[:,:chi_max], s[:chi_max], B[:chi_max,:]
        s /= np.linalg.norm(s)
        
        mps.append(A.reshape((chiL,2,-1)).transpose((1,0,2)))
        vector = (np.diag(s)@B).reshape((-1, 2**(L-i-1)))
        chiL = vector.shape[0]
    return mps

def to_vector(mps):
    L = len(mps)
    vector = mps[0]
    for i in range(1,L):
        vector = np.tensordot(vector, mps[i], [-1,1])
    return vector.reshape((2**L,))

def pad_to_umps(mps):
    L = len(mps)
    chi = jnp.max(jnp.array([i.shape for i in mps]))
    umps = []
    for i in range(L):
        _, chiL, chiR = mps[i].shape
        umps.append(jnp.pad(mps[i], ((0,0),(0,chi-chiL),(0,chi-chiR))))
    return jnp.array(umps)

