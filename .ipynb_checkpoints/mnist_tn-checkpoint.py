from typing import Mapping, Tuple, Generator, List
from absl import app

import jax
import jax.numpy as jnp
import numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]
TN = List[np.ndarray]

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

def process(batch: Batch) -> Batch:
    x = batch['image'].astype(jnp.float32)
    x = x / 255.
    nb, _, _, nc = x.shape
    x = jax.image.resize(x, [nb,14,14,nc], "nearest")
    img = jnp.concatenate([jnp.cos(x*np.pi/2.), jnp.sin(x*np.pi/2)], axis=-1)
    img = img.reshape((img.shape[0], 14*14, img.shape[3]))
    return {
        "image": img,
        "label": batch["label"],
    }

def split(tn: TN, m: int, flip: bool) -> TN:
    tnsr = tn[m]
    p0, p1, vL, vR = tnsr.shape
    tnsr = tnsr.transpose((0,2,1,3)).reshape((p0*vL, p1*vR))
    if flip:
        tnsr = tnsr.transpose((1,0,3,2))
    q, r = jnp.linalg.qr(tnsr)
    q = q.reshape((p0, vL, -1))
    r = r.reshape((-1, p1, vR)).transpose((1,0,2))

    if flip:
        r = r.transpose((0,2,1))
        q = q.transpose((0,2,1))
        q, r = r, q
    
    tn[m] = q
    tn[m+1] = r
    return tn

def join(tn: TN, m: int) -> TN:
    tn[m] = np.tensordot(tn[m], tn[m+1], [2,1]).transpose((0,2,1,3))
    tn.pop(m+1)
    return tn

def evaluate(tn: TN, batch: Batch) -> jnp.ndarray:
    # TODO: work out how to properly batch this...
    x = process(batch)["image"]
    batch_size = x.shape[0]
    L = len(tn) 
    predicts = jnp.zeros((batch_size, 10))
    
    for j in range(batch_size):
        out = jnp.array([1.])
        xi = x[j]
        m = 0
        for i in range(L-1):
            W = tn[i]
            if len(W.shape) == 3:
                tnsr = jnp.tensordot(xi[m,:], W, [-1,0])
                out = jnp.tensordot(out, tnsr, [0,0])
                m += 1
            elif len(W.shape) == 4:
                tnsr = jnp.tensordot(xi[m,:], W, [-1,0])
                tnsr = jnp.tensordot(xi[m+1,:], tnsr, [-1,0])
                out = jnp.tensordot(out, tnsr, [0,0])
                m += 2
            print(out)
        print(out.reshape(-1))
        predicts = predicts.at[j].set(out.reshape(-1))
    return predicts

def loss(tn: TN, batch: Batch) -> jnp.ndarray:
    logits = evaluate(tn, batch)
    labels = jax.nn.one_hot(batch["label"], 10)
    l2_loss = 0.5 * jnp.sum(jnp.square(logits - labels))
    return l2_loss

def init(L: int) -> List[jnp.ndarray]:
    key = jax.random.PRNGKey(42)
    tn = [jax.random.normal(key, shape=(2,1,1)) for i in range(L)]
    tn[L-1] = jax.random.uniform(key, shape=(2,1,10))
    return tn

def main(_):
    def update(tn: TN,
               opt_state: optax.OptState,
               batch: Batch,
               m: int) -> Tuple[TN, optax.OptState]:

        grads = jax.grad(loss)(tn, batch)
        print(grads)
        updates, opt_state = opt.update(grads, opt_state)

        # Here, I actually update everything, not just the joined tensor. We'll
        # see if it works...
        tn[m] += updates[m]
        print(updates[m])
        return tn, opt_state

    def accuracy(tn: TN, batch: Batch) -> jnp.ndarray:
        predictions = evaluate(tn, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    L = 14*14
    tn = init(L)
    opt = optax.sgd(0.001)
    train = load_dataset("train", is_training=True, batch_size=1)
    train_eval = load_dataset("train", is_training=False, batch_size=10000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    opt_state = opt.init(tn)

    step = 0
    dm = 1
    tn = join(tn, 0)
    m = 0

    for step in range(10):
        print(step)
        print([i.shape for i in tn])
        if step % 1000 == 0 and step > 0:
            test_accuracy = accuracy(tn, next(test_eval))
            train_accuracy = accuracy(tn, next(train_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train/Test accuracy: "
                    f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        tn, opt_state = update(tn, opt_state, next(train), m)
        tn = split(tn, m, dm==-1)
        m += dm      
        tn = join(tn, m)

        if m == L-2:
            dm = -1
        elif m == 0:
            dm = +1



if __name__ == '__main__':
    app.run(main)
