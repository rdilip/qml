import jax
import jax.numpy as jnp
from jax.config import config
from jax.scipy.linalg import expm
import torch
from itertools import product

import numpy as np
import optax
from progress.bar import Bar


config.update("jax_enable_x64", True)

pauli = [jnp.eye(2), 
         jnp.array([[0,1],[1,0]]),
         jnp.array([[0,-1.j],[1.j,0]]),
         jnp.array([[1,0],[0,-1]])]

two_site_gates = list(product(pauli, pauli))
two_site_gates = jnp.array([jnp.kron(pair[0], pair[1]) for pair in two_site_gates])
Nbasis = len(two_site_gates)

def compute_unitary_gates(params):
    return expm(-0.5j*jnp.tensordot(params, two_site_gates, [0,0]))

compute_layer_mapped = jax.vmap(compute_unitary_gates, in_axes=0, out_axes=0)
compute_mapped = jax.vmap(compute_layer_mapped, in_axes=0, out_axes=0)

def gate_list_to_matrix(gate_list):
    """ gate_list should be a array with shape (Nlayers, Ngates, 4, 4) """
    Nlayers, Ngates, _, _ = gate_list.shape
    L = Ngates+1
    dim = 2**L

    total_U = jnp.eye(dim, dtype=complex)

    for layer in range(Nlayers):
        for gate in range(Ngates):
            Uleft = jnp.kron(np.eye(2**gate), gate_list[layer, gate])
            U = jnp.kron(Uleft, np.eye(2**(L-gate-2)))
            total_U = total_U @ U
    return total_U

@jax.jit
def loss(params, inp, out):
    gate_list = compute_mapped(params)
    total_U = gate_list_to_matrix(gate_list)
    return 1. - jnp.real(inp @ out.conj())


def main():
    opt = optax.adam(1.e-4)
    
    def update(params, opt_state, inp, out):
        grads = jax.grad(loss)(params, inp, out)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    Nruns = 10000
    Nlayers = 1
    Ngates = 10
    Nqubits = 11
    inp = np.zeros((2**Nqubits), dtype=complex)
    inp[0] = 1.
    inp = jnp.array(inp)

    out = jnp.array(np.array(torch.load("vector.pt")))
    params = jnp.array(np.random.rand(Nlayers, Ngates, 16))
    opt_state = opt.init(params)

    bar = Bar("Number of runs: ", max=Nruns)
    for i in range(Nruns):
        params, opt_state = update(params, opt_state, inp, out)
        bar.next()
    breakpoint()
    bar.finish()
    torch.save(params, "params.pt")


if __name__ == '__main__':
    main()
