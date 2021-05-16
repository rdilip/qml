from typing import Mapping, Tuple, Generator, List, Callable
from absl import app

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import optax

import tensorflow as tf
import tensorflow_datasets as tfds
from itertools import product

from progress.bar import Bar

Circuit = List[List[List]]
MPS     = List[jnp.ndarray]

pauli = [jnp.eye(2), 
         jnp.array([[0,1],[1,0]]),
         jnp.array([[0,-1.j],[1.j,0]]),
         jnp.array([[1,0],[0,-1]])]

two_site_gates = list(product(pauli, pauli))
two_site_gates = jnp.array([jnp.kron(pair[0], pair[1]) for pair in two_site_gates])
Nbasis = len(two_site_gates)

def compute_unitary_gates(params):
    if len(params) == 0:
        return jnp.array([], dtype=complex)
    return expm(-0.5j*jnp.tensordot(params, two_site_gates, [0,0])).reshape((2,2,2,2))

def init_brickwall(L: int, Nlayers: int) -> Circuit:
    circuit = []
    for layer in range(Nlayers):
        single_layer = []
        m = layer % 2
        for i in range(L-1):
            if (i+m) % 2 == 0:
                single_layer.append(jnp.zeros(Nbasis))
            else:
                single_layer.append(jnp.array([]))
        circuit.append(single_layer)
    return circuit

def init_mps(L: int) -> MPS:
    return [jnp.array([1.,0.]).reshape((2,1,1)) for i in range(L)]

def split_truncate_theta(theta: jnp.ndarray,
                         chi_max: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    chiL, p0, p1, chiR = theta.shape
    theta = theta.reshape((chiL*p0, chiR*p1))
    U, s, V = jnp.linalg.svd(theta, full_matrices=False)
    U, s, V = U[:, :chi_max], s[:chi_max], V[:chi_max, :]
    s /= jnp.linalg.norm(s)

    V = jnp.diag(s) @ V

    X = U.reshape(chiL, p0, -1).transpose((1,0,2))
    Y = V.reshape((-1, p1, chiR)).transpose((1,0,2))
    return X, Y

def _apply_U(B1: jnp.ndarray, B2: jnp.ndarray, U: jnp.ndarray, chi_max: int):
    theta = jnp.tensordot(B1, B2, [-1,-2])
    Utheta = jnp.tensordot(U, theta, [[0,1],[0,2]]).transpose((2,0,1,3))
    return split_truncate_theta(Utheta, chi_max)

def apply_U(Psi: MPS, U: jnp.ndarray, i: int, chi_max: int) -> MPS:
    X, Y = _apply_U(Psi[i], Psi[i+1], U, chi_max)
    Psi[i] = X
    Psi[i+1] = Y
    return Psi

def shift_ortho_center(Psi: MPS, i: int, j: int, chi_max: int) -> MPS:
    """ From i to j """
    reverse = i > j
    L = len(Psi)
    if reverse:
        Psi = [psi.transpose((0,2,1)) for psi in Psi[::-1]]
        i, j = L-1-i, L-1-j
    for k in range(i, j):
        theta = jnp.tensordot(Psi[k], Psi[k+1], [2,1]).transpose((1,0,2,3))
        X, Y = split_truncate_theta(theta, chi_max=chi_max)
        Psi[k] = X
        Psi[k+1] = Y
    if reverse:
        Psi = [psi.transpose((0,2,1)) for psi in Psi[::-1]]
    return Psi

def apply_circuit(Psi: MPS, circuit: Circuit, chi_max: int) -> MPS:
    Nlayers = len(circuit)
    L       = len(circuit[0])

    ucircuit = jax.tree_util.tree_map(compute_unitary_gates,
                                      circuit, 
                                      is_leaf=lambda x: hasattr(x, 'dtype'))

    m = 0 # ortho center
    for layer in range(Nlayers):
        for i in range(L-1):
            if circuit[layer][i].size:
                Psi = shift_ortho_center(Psi, m, i, chi_max)
                Psi = apply_U(Psi, ucircuit[layer][i], i, chi_max)
                m = i
    return Psi



