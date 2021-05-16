from typing import Generator, Mapping, Tuple
from absl import app

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import tensorflow_datasets as tfds

from progress.bar import Bar

Batch = Mapping[str, np.ndarray]

def net_fn(batch: Batch) -> jnp.ndarray:
    x = batch["image"].astype(jnp.float32) / 255.
    #mlp = hk.Sequential([
    #    hk.Flatten(),
    #    hk.Linear(200), jax.nn.relu,
    #    hk.Linear(100), jax.nn.relu,
    #    hk.Linear(10),
    #])
    mlp = hk.Sequential([
            hk.Conv2D(20, (5,5), stride=1), jax.nn.relu,
            hk.MaxPool((2,2)),
            hk.Linear(100), jax.nn.relu,
            hk.Linear(10,
        ])
    return mlp(x)

def load_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
        ) -> Generator[Batch, None, None]:
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def main(_):
    # Make network and optimizer
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(1.e-3)

    # Training loss (cross-entropy)
    @jax.jit
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """ Compute loss of network """
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1.e-4 * l2_loss
    
    # Classification accuracy
    @jax.jit
    def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    batch_size = 50
    Nepochs = 20

    train = load_dataset("train", is_training=True, batch_size=batch_size)
    train_eval = load_dataset("train", is_training=False, batch_size=10000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)


    params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
    opt_state = opt.init(params)

    for epoch in range(Nepochs):
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=60000//batch_size)
        for batch in range(60000//batch_size):
            params, opt_state = update(params, opt_state, next(train))
            avg_params = ema_update(params, avg_params)
            bar.next()
        bar.finish()

        train_accuracy = accuracy(avg_params, next(train_eval))
        test_accuracy = accuracy(avg_params, next(test_eval))
        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
        print(f"Train / Test accuracy: "
                f"{train_accuracy:.4f} / {test_accuracy:.4f}.")

if __name__ == '__main__':
    app.run(main)

