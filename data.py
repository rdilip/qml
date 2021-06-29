""" Handles all data loading and preprocessing """

from typing import Generator, Tuple, Mapping, Callable
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from compress import to_mps

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
    
    if kernel is None:
        kernel = lambda x: jnp.concatenate([1-x, x], axis=-1)
    
    nb, _, _, nc = x.shape
    x = np.array(jax.image.resize(x, [nb,resize[0],resize[1],nc], "nearest"))
    
    if compress:
        img = [img.reshape(np.prod(resize)) for img in x]
        img = pad_to_umps(to_mps(im, compress_kwargs['chi']) for im in img])
    else:
        img = kernel(x)
        img = img.reshape((nb, 2, resize[0]*resize[1]//2, 2))

    return {
        "image": img,
        "label": batch["label"],
    }


