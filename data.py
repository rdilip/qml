""" Handles all data loading and preprocessing """

from typing import Generator, Tuple, Mapping, Callable
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from compress import to_mps, pad_to_umps

Batch = Mapping[str, np.ndarray]


def load_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
        labels: Tuple[int, int]=None
        ) -> Generator[Batch, None, None]:
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if labels is not None:
        ds = ds.filter(lambda fd: tf.logical_or(fd['label'] == labels[0], fd['label'] == labels[1]))
    if is_training:
        ds = ds.shuffle(10*batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def process_img(batch: Batch,
                resize: Tuple,
                kernel: Callable=None,
                compress=False,
                **compress_kwargs) -> Batch:
    """ Args:
            kernel: Function that accepts an array of size (nb, w, h, 1) and
            outputs an array of size (nb, w, h, nc). nc is the number of
            channels, typically 2.
    """
    x = batch['image'].astype(jnp.float32)
    assert (resize[0] * resize[1]) % 2 == 0
    x = x / 255.
    L = np.prod(resize)
    
    if kernel is None:
        kernel = lambda x: jnp.concatenate([1-x, x], axis=-1)
    
    nb, _, _, nc = x.shape
    x = np.array(jax.image.resize(x, [nb,resize[0],resize[1],nc], "nearest"))
    
    # TODO: add option for kernel on top of added qubits -- Fourier?
    if compress:
        chi = compress_kwargs['chi']
        x = x.reshape((nb, L))
        img = [pad_to_umps(to_mps(v, chi_max=chi)) for v in x]
        img = jnp.array(img)
        mode = compress_kwargs.getdefault("mode", None)
        qa = compress_kwargs.getdefault("add_qubits", 0)
        nb, L, d, chi, _ = img.shape

        if mode == "interleave":
            img_qa = jnp.zeros((nb,(qa+1)*L,d,chi,chi))
            img_qa = jax.ops.index_update(img_qa, (slice(nb),slice(0,(qa+1)*L,(qa+1))), img)
            A = jnp.array([jnp.eye(chi)]*d)

            for i in range(qa):
                img_qa = jax.ops.index_update(img_qa, (slice(nb),slice(i+1,(qa+1)*L,qa+1)), A)
            img = img_qa
        elif mode == "append":
            A = jnp.array([jnp.eye(chi)]*d*qa*nb).reshape((nb,qa,d,chi,chi))
            img = jnp.concatenate((img, A), axis=1)
    else:
        img = kernel(x)
        img = img.reshape((nb, L, img.shape[-1]))

    return {
        "image": img,
        "label": batch["label"],
    }


