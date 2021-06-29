# coding: utf-8
import numpy as np
import jax.numpy as jnp

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

