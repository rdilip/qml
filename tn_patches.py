# !/usr/bin/env python
""" MPS learning with patchwork over image """
from typing import Tuple 

import jax
import jax.numpy as jnp
import numpy as np

import optax
import time
from progress.bar import Bar

from data_pt import *
from data_tracker import DataTracker
from itertools import cycle

Batch = Tuple[jnp.ndarray, int]

@jax.jit
def _reduce_mps(x0, x1):
    """ Helper function for jax.lax.scan """
    return (jnp.tensordot(x0, x1, [[2,3],[0,1]]), x1)

@jax.jit
def mps_on_img(tn, img):
    center = jnp.einsum("xpab,xpcd->xacbd", tn['center'], img[1:-1])
    lbound = jnp.tensordot(tn['lbndry'], img[0], [0,0]).transpose((0,2,1,3))
    rbound = jnp.tensordot(tn['rbndry'], img[-1], [0,0]).transpose((0,2,1,3))
    contracted = jax.lax.scan(_reduce_mps, lbound, center)[0]
    contracted = _reduce_mps(contracted, rbound)[0]
    return contracted

@jax.jit
def evaluate(tn, patched_img):
    Npatches, Lpp, _, _, chi = patched_img.shape
    contractions = []
    for i in range(Npatches):
        contractions.append(mps_on_img(tn[i], patched_img[i]))
    contractions = jnp.array(contractions)
    end_contraction = jax.lax.scan(_reduce_mps, contractions[0], contractions[1:])[0]
    pred = jnp.tensordot(end_contraction[0,0,:,0], tn[-1], [0,0])
    return pred

evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn, batch: Batch) -> jnp.ndarray:
    logits = evaluate_batched(tn, batch[0])
    labels = jax.nn.one_hot(batch[1], 10)
    
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    
    return softmax_xent

# TN initializers
def _init_subnetwork(L: int, d: int, chi: int) -> jnp.ndarray:
    w = jnp.stack(d * L * [jnp.eye(chi)])
    w = w.reshape((L, d, chi, chi))
    return w + np.random.normal(0, 1.e-2, size=w.shape)

def init_mps(L: int, chi: int):
    tn = {}
    tn['lbndry'] = _init_subnetwork(1, 2, chi)[0,:,0,:].reshape((2,1,chi))
    tn['rbndry'] = _init_subnetwork(1, 2, chi)[0,:,:,0].reshape((2,chi,1))
    tn['center'] = _init_subnetwork(L-2, 2, chi)

    return tn

def init_network(Lpc: int, Npatches: int, chi: int):
    """ Initializes a larger network of patchwise classifiers. L is the length
    per classifier. """
    tn = []
    for i in range(Npatches):
        tn.append(init_mps(Lpc, chi))
    tn.append(_init_subnetwork(1, 10, 1)[:, :, 0, 0])
    return tn

@jax.jit
def accuracy(tn, batch: Batch) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch[0])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch[1])

def main(chi_tn, chi_img, pd):
    opt = optax.adam(1.e-4) # High rate for fudged contraction -- will fix.

    @jax.jit
    def update(tn,
               opt_state: optax.OptState,
               batch: Batch):
        grads = jax.grad(loss)(tn, batch)
        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        return tn, opt_state

    resize = (32,32)
    Lpc = int(np.ceil(np.log2(np.prod(resize) / np.prod(pd)))) + 1 # 2 channels
    tn = init_network(Lpc, np.prod(pd), chi_tn)

    batch_size = 50
    Nepochs = 30
    
    training_generator = load_training_set(batch_size=batch_size,
                                             resize=resize, 
                                             chi_max=chi_img,
                                             patch_dim=pd)
    train_eval, test_eval = load_eval_set(batch_size=1000,
                                         resize=resize, 
                                         chi_max=chi_img,
                                         patch_dim=pd)
    train_eval, test_eval = cycle(train_eval), cycle(test_eval)
    
    opt_state = opt.init(tn)
    losses = []
    attr = ["raw", "gpu", "mnist", f"{resize[0]}x{resize[1]}", f"split_{pd[0]}x{pd[1]}", f"chi_img{chi_img}"]
    prepend = f"chi{chi_tn}"

    dt = DataTracker(attr, prepend=prepend)
    test_accuracy = accuracy(tn, next(test_eval))
    train_accuracy = accuracy(tn, next(train_eval))
    train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
    start = time.time()

    dt.register("loss", lambda: loss(tn, next(test_eval)))
    dt.register("train_accuracy", lambda: train_accuracy)
    dt.register("test_accuracy", lambda: test_accuracy)
    dt.register("time_elapsed", lambda: time.time() - start)
    dt.register("model", lambda: tn)

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
    main(2, 2, (2, 2))
    
