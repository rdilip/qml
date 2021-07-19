""" TN learning using a simplified method of Google's paper. There is nontrivial bond
dimension on the side wings which is dumb """

from typing import Mapping, List

import jax
import jax.numpy as jnp
import numpy as np

import optax
import time
from progress.bar import Bar

from data_pt import *
from data_tracker import DataTracker
from itertools import cycle

Batch = Mapping[str, np.ndarray]
TN = Mapping[str, jnp.ndarray]

@jax.jit
def _reduce_mps(x0, x1):
    return (jnp.tensordot(x0, x1, [[2,3],[0,1]]), x1)

@jax.jit
def evaluate(tn, img):
    """ img should be shape (L, 2) """

    chi = tn['left'].shape[-1]

    L = img.shape[0]
    start = jnp.tensordot(tn['left'].reshape((-1,1,chi)), img[0], [0,0]).transpose((0,2,1,3))
    center = jnp.einsum("xpab,xpcd->xacbd", tn['center'], img[1:-1])
    center, _ = jax.lax.scan(_reduce_mps, start, center)

    pred = jnp.tensordot(tn['right'], img[-1], [0,0]).transpose((0,2,1,3))
    pred = _reduce_mps(center, pred)[0].ravel()
    return pred
  
evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn: TN, batch: Batch) -> jnp.ndarray:
    logits = evaluate_batched(tn, batch[0])
    labels = jax.nn.one_hot(batch[1], 10)
    
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    
    return softmax_xent

def _init_subnetwork(L: int, d: int, chi: int) -> jnp.ndarray:
    w = jnp.stack(d * L * [jnp.eye(chi)])
    w = w.reshape((L, d, chi, chi))
    return w + np.random.normal(0, 1.e-2, size=w.shape)

def init(L: int, chi: int, Nlabels: int=10) -> TN:
    tn = {}
    tn['left'] = _init_subnetwork(1,2,chi)[0,:,0,:]
    tn['right'] = _init_subnetwork(Nlabels, 2, chi)[...,0].transpose((1,2,0))
    tn['center'] = _init_subnetwork(L-2, 2, chi)

    return tn

@jax.jit
def accuracy(tn: TN, batch: Batch) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch[0])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch[1])

def main():
    opt = optax.adam(1.e-4) # High rate for fudged contraction -- will fix.

    @jax.jit
    def update(tn: TN,
               opt_state: optax.OptState,
               batch: Batch) -> Tuple[TN, optax.OptState]:
        grads = jax.grad(loss)(tn, batch)
        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        return tn, opt_state

    shape = (32,32)
    L = shape[0]*shape[1]
    chi = 8
    tn = init(L, chi)
    pd = (1,1)
    chi_img=1
    batch_size = 50
    Nepochs = 100


    training_generator = load_training_set(batch_size=batch_size,
                              resize=shape, patch_dim=pd, chi_max=chi_img)
    train_eval, test_eval = load_eval_set(batch_size=100, resize=shape, patch_dim=pd, chi_max=chi_img)
    train_eval, test_eval = cycle(train_eval), cycle(test_eval)

    opt_state = opt.init(tn)
    losses = []
    attr = ["raw", "gpu", "mnist", "product", f"{shape[0]}x{shape[1]}"]
    prepend = f"chi{chi}"
    dt = DataTracker(attr, prepend=prepend)

    test_accuracy = accuracy(tn, next(test_eval))
    train_accuracy = accuracy(tn, next(train_eval))
    train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
    start = time.time()

    dt.register("loss", lambda: loss(tn, next(test_eval)))
    dt.register("train_accuracy", lambda: train_accuracy)
    dt.register("test_accuracy", lambda: test_accuracy)
    dt.register("time_elapsed", lambda: time.time() - start)

    for epoch in range(Nepochs):
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=60000//batch_size)
        for batch in training_generator:
            tn, opt_state = update(tn, opt_state, batch)
            bar.next()
        bar.finish()
        
        test_accuracy = accuracy(tn, next(test_eval))
        train_accuracy = accuracy(tn, next(train_eval))
        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
        print(f"Train/Test accuracy: "
                f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        dt.update(save_interval=1)
    dt.save()

if __name__ == '__main__':
    main()

