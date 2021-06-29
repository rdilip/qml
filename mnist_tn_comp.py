""" TN learning using a simplified method of Google's paper. There is nontrivial bond
dimension on the side wings which is dumb """

from typing import Mapping, List

import jax
import jax.numpy as jnp
import numpy as np

import optax
import time
from progress.bar import Bar

from data import *
from data_tracker import DataTracker

Batch = Mapping[str, np.ndarray]
TN = Mapping[str, jnp.ndarray]

@jax.jit
def _reduce_mps(x0, x1):
    """ Helper function for jax.lax.scan """
    return (jnp.tensordot(x0, x1, [[2,3],[0,1]]), x1)

@jax.jit
def evaluate(tn, img_mps):
    L = img_mps.shape[0]

    left = jnp.einsum("xpab,xpcd->xacbd", tn['left'], img_mps[1:L//2])
    left = jax.lax.scan(_reduce_mps, left[0], left[1:])[0]
    right = jnp.einsum("xpab,xpcd->xacbd", tn['right'], img_mps[L//2:L-1])
    right = jax.lax.scan(_reduce_mps, right[0], right[1:])[0]

    lbound = jnp.tensordot(tn['left_boundary'], img_mps[0][:,0,:], [0,0])
    rbound = jnp.tensordot(tn['right_boundary'], img_mps[L-1][:,:,0], [0,0])

    left = jnp.tensordot(lbound, left, [[0,1],[0,1]])
    right = jnp.tensordot(right, rbound, [[2,3],[0,1]])
    env = jnp.tensordot(left, right, [1,1])
    pred = jnp.tensordot(tn['center'], env, [[1,2],[0,1]])
    return pred
  
evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn: TN, batch: Batch) -> jnp.ndarray:
    logits = evaluate_batched(tn, batch["image"])
    labels = jax.nn.one_hot(batch["label"], 10)
    
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    
    return softmax_xent

def init(L: int, chi: int) -> TN:
    tn = {}

    def _init_subnetwork(L: int, d: int) -> jnp.ndarray:
        w = jnp.stack(d * L * [jnp.eye(chi)])
        w = w.reshape((L, d, chi, chi))
        return w + np.random.normal(0, 1.e-2, size=w.shape)

    tn['left_boundary'] = _init_subnetwork(1, 2)[0,:,:,0]
    tn["left"] = _init_subnetwork(L//2-1, 2)
    tn["center"] = _init_subnetwork(1, 10)[0] # this is num labels
    tn["right"] = _init_subnetwork(L-L//2-1, 2)
    tn['right_boundary'] = _init_subnetwork(1, 2)[0,:,0,:]

    return tn

@jax.jit
def accuracy(tn: TN, batch: Batch) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch["image"])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

def main(chi, chi_img, mode, Nqubits, dataset="mnist:3.*.*"):
    opt = optax.adam(1.e-4) # High rate for fudged contraction -- will fix.

    @jax.jit
    def update(tn: TN,
               opt_state: optax.OptState,
               batch: Batch) -> Tuple[TN, optax.OptState]:
        grads = jax.grad(loss)(tn, batch)
        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        return tn, opt_state

    shape = (28,28)

    L = int(np.ceil(np.log2(shape[0]*shape[1])))

    if mode == "interleave":
        L *= (Nqubits+1)
    elif mode == "append":
        L += Nqubits

    tn = init(L, chi)
    batch_size = 50
    Nepochs = 100

    process = lambda x: process_img(x, 
                                    shape,
                                    None,
                                    compress=True,
                                    chi=chi_img,
                                    mode=mode,
                                    add_qubits=Nqubits)


    train = load_dataset("train", is_training=True, batch_size=batch_size)
    train_eval = load_dataset("train", is_training=False, batch_size=1000)
    test_eval = load_dataset("test", is_training=False, batch_size=1000)

    #batch = process(next(test_eval))
    #print(batch['image'].shape)


    opt_state = opt.init(tn)
    losses = []
    attr = ["raw", "cpu", dataset.split(":")[0], f"chi_img{chi_img}", f"{shape[0]}x{shape[1]}",\
            mode, f"Nqubits{Nqubits}"]

    prepend = f"chi{chi}"
    dt = DataTracker(attr, prepend=prepend)

    test_accuracy = accuracy(tn, process(next(test_eval)))
    train_accuracy = accuracy(tn, process(next(train_eval)))
    train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
    start = time.time()

    dt.register("loss", lambda: loss(tn, process(next(test_eval)))) 
    dt.register("train_accuracy", lambda: train_accuracy)
    dt.register("test_accuracy", lambda: test_accuracy)
    dt.register("time_elapsed", lambda: time.time() - start)

    for epoch in range(Nepochs):
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=60000//batch_size)
        for batch in range(60000//batch_size):
            tn, opt_state = update(tn, opt_state, process(next(train)))
            bar.next()
        bar.finish()
        
        test_accuracy = accuracy(tn, process(next(test_eval)))
        train_accuracy = accuracy(tn, process(next(train_eval)))
        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
        print(f"Train/Test accuracy: "
                f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        dt.update(save_interval=10)
    dt.save()

if __name__ == '__main__':
    main(10,4,"interleave",1)
    # main: chi, chi_img, mode, Nqubits, dataset="mnist:3.*.*"

