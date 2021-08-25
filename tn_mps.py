""" TN learning using a simplified method of Google's paper. There is nontrivial bond
dimension on the side wings which is dumb """

import jax
from jax.ops import index_update, index
import jax.numpy as jnp
from jax import lax 
from jax.experimental.optimizers import l2_norm
import optax

from typing import Mapping, List
import numpy as np
import time
from progress.bar import Bar
from itertools import cycle

from data_pt import *
from data_tracker import DataTracker, round_sf

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
    pred = _reduce_mps(center, pred)[0][0,0,:,0].ravel()
    return pred
  
evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jax.jit
def loss(tn: TN, batch: Batch) -> jnp.ndarray:
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

def main(pd, chi_tn, chi_img, 
            Nepochs=300, 
            dataset="mnist",
            batch_size=128,
            eval_size=1000):
    opt = optax.adam(lr) 

    @jax.jit
    def update(tn: TN,
               opt_state: optax.OptState,
               batch: Batch) -> Tuple[TN, optax.OptState]:
        grads = jax.grad(loss)(tn, batch)

        flattened_grads = jax.flatten_util.ravel_pytree(grads)[0]
        max_grad, min_grad, mean_grad, var_grad = jnp.max(flattened_grads),\
                                                  jnp.min(flattened_grads),\
                                                  jnp.mean(flattened_grads),\
                                                  jnp.var(flattened_grads)
        max_grad, min_grad, mean_grad, var_grad = jax.device_get(\
                (max_grad, min_grad, mean_grad, var_grad))

        # scale = lax.select(jnp.abs(mean_grad) < 1., min_grad, max_grad)
        scale = 1
        grads = jax.tree_map(lambda x: x / scale, grads)

        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        return tn, opt_state, max_grad, mean_grad, var_grad

    shape = (32,32)
    Npatches = int(np.prod(shape) / np.prod(pd))
    if not check_param_saturation(shape, pd, chi_img):
        raise ValueError("chi_img is not compressing")

    L = Npatches * int(np.ceil(np.log2(np.prod(pd))) + 1)
    tn = init(L, chi_tn)
    cache_transformed_dataset(dataset_name=dataset, resize=shape, chi_max=chi_img, patch_dim=pd)

    training_generator = load_training_set(dataset_name=dataset,
            batch_size=batch_size, resize=shape, patch_dim=pd, chi_max=chi_img)
    train_eval, test_eval = load_eval_set(
                                        dataset_name=dataset,
                                        batch_size=eval_size,
                                        resize=shape,
                                        patch_dim=pd,
                                        chi_max=chi_img)
    train_eval, test_eval = cycle(train_eval), cycle(test_eval)

    opt_state = opt.init(tn)
    attr = ["raw", "cpu", dataset_name, f"size_{shape[0]}x{shape[1]}", f"patch_{pd[0]}x{pd[1]}",\
                f"chi_img{chi_img}"]

    prepend = f"chi{chi_tn}"
    if lr != 1.e-4:
        prepend = f"chi{chi_tn}_lr{lr}"

    dt = DataTracker(attr, prepend=prepend, experimental=False, overwrite=False)

    test_accuracy = accuracy(tn, next(test_eval))
    train_accuracy = accuracy(tn, next(train_eval))
    train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
    start = time.time()

    losses = dt.register("loss", lambda: loss(tn, next(test_eval)))
    dt.register("train_accuracy", lambda: train_accuracy)
    dt.register("test_accuracy", lambda: test_accuracy)
    dt.register("time_elapsed", lambda: time.time() - start)
    tn = dt.register("model", lambda: tn)[-1]

    print("Starting loss: " + str(loss(tn, next(test_eval))))

    for epoch in range(Nepochs):
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=60000//batch_size)
        max_grads, mean_grads, var_grads = [], [], []
        for batch in training_generator:
            tn, opt_state, max_grad, mean_grad, var_grad = update(tn, opt_state, batch)

            max_grads.append(max_grad)
            mean_grads.append(mean_grad)
            var_grads.append(var_grad)
            bar.next()
        bar.finish()
       
        test_accuracy = accuracy(tn, next(test_eval))
        train_accuracy = accuracy(tn, next(train_eval))
        test_loss = loss(tn, next(test_eval))
        train_loss = loss(tn, next(train_eval))

        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
        train_loss, test_loss = jax.device_get((train_loss, test_loss))
        print(f"Train/Test accuracy: "
                f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        print(f"Train/Test loss: "
                f"{train_loss}/{test_loss}")
        print([np.mean(g) for g in [max_grads, mean_grads, var_grads]])
        dt.update(save_interval=1)

    dt.save()

