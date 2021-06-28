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
def evaluate(tn: TN, img: jnp.ndarray) -> jnp.ndarray:
    # TODO: the einsum is EXPENSIVE -- figure out a better way to do this.
    l, r, center = tn['left'], tn['right'], tn['center']
    l = jnp.einsum("ndab,nd->nab", l, img[0])
    r = jnp.einsum("ndab,nd->nab", r, img[1])

    f = lambda a, b: (jnp.matmul(a, b), b)
    l, _ = jax.lax.scan(f, jnp.eye(l.shape[1]), l)
    r, _ = jax.lax.scan(f, jnp.eye(r.shape[1]), r)

    pred = jnp.tensordot(l, center, [1,1])
    pred = jnp.tensordot(pred, r, [[0,2],[1,0]])
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
    
    tn["left"] = _init_subnetwork(L//2, 2)
    tn["right"] = _init_subnetwork(L-L//2, 2)
    tn["center"] = _init_subnetwork(1, 10)[0] # this is num labels

    return tn

@jax.jit
def accuracy(tn: TN, batch: Batch) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch["image"])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

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

    shape = (28,28)
    L = shape[0]*shape[1]
    chi = 10
    tn = init(L, 10)
    batch_size = 50
    Nepochs = 120

    process = lambda x: process_img(x, shape, None)

    train = load_dataset("train", is_training=True, batch_size=batch_size)
    train_eval = load_dataset("train", is_training=False, batch_size=1000)
    test_eval = load_dataset("test", is_training=False, batch_size=1000)

    opt_state = opt.init(tn)
    losses = []
    attr = ["raw","gpu","mnist","tn_center","uncompressed", f"chi{chi}", f"{shape[0]}x{shape[1]}"]
    dt = DataTracker(attr)

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
        dt.update(save_interval=1)
    dt.save()

if __name__ == '__main__':
    main()
