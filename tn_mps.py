""" TN learning using stochastic gradient descent. """

import jax
import jax.numpy as jnp
from jax import lax 
import numpy as np


@jax.jit
def _reduce_mps(x0, x1):
    comb = jnp.tensordot(x0, x1, [[2,3],[0,1]])
    comb /= jnp.linalg.norm(comb)
    return (comb, x1)

@jax.jit
def evaluate(tn, img):
    """ img should be shape (L, 2) """
    chi = tn['left'].shape[-1]

    L = img.shape[0]
    start = jnp.tensordot(tn['left'].reshape((-1,1,chi)), img[0], [0,0]).transpose((0,2,1,3))
    center = jnp.einsum("xpab,xpcd->xacbd", tn['center'], img[1:-1])
    center, _ = jax.lax.scan(_reduce_mps, start, center)

    pred = jnp.tensordot(tn['right'], img[-1], [0,0]).transpose((0,2,1,3))
    pred = _reduce_mps(center, pred)[0][0,0,:,0].ravel()
    return pred
  
evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn, batch) -> jnp.ndarray:
    logits = evaluate_batched(tn, batch[0])
    labels = jax.nn.one_hot(batch[1], 10)
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(tn))
    
    return softmax_xent + 1.e-4 * l2_loss

def _init_subnetwork(L: int, d: int, chi: int) -> jnp.ndarray:
    w = jnp.stack(d * L * [jnp.eye(chi)])
    w = w.reshape((L, d, chi, chi))
    return w + np.random.normal(0, 1.e-4, size=w.shape)

def init(L: int, chi: int, Nlabels: int=10):
    tn = {}
    tn['left'] = _init_subnetwork(1,2,chi)[0,:,0,:]
    tn['right'] = _init_subnetwork(Nlabels, 2, chi)[...,0].transpose((1,2,0))
    tn['center'] = _init_subnetwork(L-2, 2, chi)

    return tn

@jax.jit
def accuracy(tn,  batch ) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch[0])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch[1])

