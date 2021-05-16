""" Implements MNIST learning as described in Stoudenmire Schwab. """

from typing import Mapping, Tuple, Generator, List, Callable
from absl import app

import jax
import jax.numpy as jnp
import numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

from progress.bar import Bar

Batch = Mapping[str, jnp.ndarray]
TN = Mapping[str, jnp.ndarray]

# Data processing

def load_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int
        ) -> Generator[Batch, None, None]:
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10*batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def process_img(batch: Batch, 
                resize: Tuple,
                invert: bool,
                kernel: Callable=None) -> Batch:
    """ Args:
            kernel: Function that accepts an array of size (nb, w, h, 1) and
            outputs an array of size (nb, w, h, nc). nc is the number of
            channels, typically 2.
    """
    x = batch['image'].astype(jnp.float32)
    assert (resize[0] * resize[1]) % 2 == 0
    x = x / 255.
    
    if kernel is None:
        kernel = lambda x: jnp.concatenate([1-x, x], axis=-1)
    
    nb, _, _, nc = x.shape
    x = jnp.array(jax.image.resize(x, [nb,resize[0],resize[1],nc], "nearest"))
    img = kernel(x)
    img = img.reshape((nb, resize[0]*resize[1], 2))

    if invert:
        img = img[:, ::-1, :]

    return {
        "image": img,
        "label": batch["label"],
    }

# MPS control

@jax.jit
def split(center, chi_max=None):
    assert len(center.shape) == 5
    p0, p1, l, chiL, chiR = center.shape
    center = center.transpose((0,3,1,2,4))

    center = center.reshape((p0*chiL, p1*l*chiR))
    u, s, v = jnp.linalg.svd(center, full_matrices=False)
    if chi_max is not None:
        u, s, v = u[:, :chi_max], s[:chi_max], v[:chi_max, :]

    u = u.reshape((p0, chiL, -1))
    s = (jnp.diag(s) @ v).reshape((-1, p1, l, chiR)).transpose((1,2,0,3))
    return u, s

@jax.jit
def combine(u, v):
    return jnp.tensordot(u, v, [-1, -2]).transpose((0,3,1,2,4))

@jax.jit
def step(tn):
    assert len(tn['right']) != 0

    u, v = split(tn['center'])
    tn['left'].append(u)
    tn['center'] = combine(v, tn['right'][0])
    _ = tn['right'].pop(0)

    return tn

def init(L: int, Nlabels: int=3) -> TN:
    tn = {}
    chi = 1
    assert L > 2
    def _init_subnetwork(L: int, d: int) -> jnp.ndarray:
        w = jnp.stack(d * L * [jnp.eye(chi)])
        w = w.reshape((L, d, chi, chi))
        return w + jnp.array(np.random.normal(0, 1.e-2, size=w.shape))
    
    tn["left"] = []
    tn["right"] = [i for i in _init_subnetwork(L-2, 2)]
    tn["center"] = _init_subnetwork(Nlabels, 2*2).reshape((2, 2, Nlabels, chi, chi))

    return tn

def invert_mps(tn):
    right = [i.transpose((0,2,1)) for i in tn['left'][::-1]]
    left = [i.transpose((0,2,1)) for i in tn['right'][::-1]]
    center = tn['center'].transpose((1,0,2,4,3))
    return {'left': left, 'center': center, 'right': right}

# Evaluation
@jax.jit
def contract_mps(mps: List, img: List) -> jnp.ndarray:
    """ This contracts a full MPS, not the TN object """
    out = jnp.tensordot(mps[0], img[0], [0,0])
    for i in range(1, len(mps)):
        A = jnp.tensordot(mps[i], img[i], [0,0])
        out = jnp.tensordot(out, A, [-1, -2])
    return out

@jax.jit
def _evaluate_img(center, img, lenv=None, renv=None) -> jnp.ndarray:
    """ Evaluates an MPS given the cneter tensor and the left and right wings """
    if lenv is None:
        lenv = jnp.eye(center.shape[-2])
    if renv is None:
        renv = jnp.eye(center.shape[-1])

    output = jnp.tensordot(center, img[0], [0,0])
    output = jnp.tensordot(output, img[1], [0,0])
    output = jnp.tensordot(output, renv, [-1,0])
    output = jnp.tensordot(output, lenv, [-2,-1])
    return output.reshape(center.shape[2])

@jax.jit
def evaluate(tn: TN, img: List) -> jnp.ndarray:
    lenv, renv = None, None
    m = len(tn['left'])
    if len(tn['left']) > 0:
        lenv = contract_mps(tn['left'], img[:m])
    if len(tn['right']) > 0:
        renv = contract_mps(tn['right'], img[m+2:])
    return _evaluate_img(tn['center'], img[m:m+2], renv=renv, lenv=lenv)

evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn: TN, batch: Batch) -> jnp.ndarray:
    logits = evaluate_batched(tn, batch["image"])
    labels = jax.nn.one_hot(batch["label"], 10)
    
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    
    return softmax_xent

@jax.jit
def accuracy(tn: TN, batch: Batch) -> jnp.ndarray:
    predictions = evaluate_batched(tn, batch["image"])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

def main(_):
    opt = optax.adam(1.e-4) 
    
    @jax.jit
    def update(tn: TN,
               opt_state: optax.OptState,
               batch: Batch) -> Tuple[TN, optax.OptState]:
        grads = jax.grad(loss)(tn, batch)
        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        print("all updates applied")
        return tn, opt_state

    # shape = (28,28)
    shape = (14,14)
    L = shape[0]*shape[1]
    chi_max = 20
    Nepochs = 50
    tn = init(L, 10)

    batch_size = 50

    process = lambda x, invert: process_img(x, shape, invert, None)

    train = load_dataset("train", is_training=True, batch_size=batch_size)
    train_eval = load_dataset("train", is_training=False, batch_size=10000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    opt_state = opt.init(tn)

    invert = False
    for epoch in range(Nepochs):
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=60000//batch_size)
        for batch in range(60000//batch_size):
            tn, opt_state = update(tn, opt_state, process(next(train), invert))

            tn = step(tn)
            opt_state = opt.init(tn)
            if len(tn['right']) == 0:
                tn = invert_mps(tn)
                invert = not invert
            bar.next()
        bar.finish()
        
        test_accuracy = accuracy(tn, process(next(test_eval), invert))
        train_accuracy = accuracy(tn, process(next(train_eval), invert))
        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))

        print(f"Train/Test accuracy: "
                f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        losses.append(loss(tn, process(next(test_eval))))

if __name__ == '__main__':
    app.run(main)
