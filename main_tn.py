import jax
import jax.numpy as jnp
import optax

import numpy as np
import time
from progress.bar import Bar
from itertools import cycle

from tn_mps import *

from torch.utils.data import DataLoader

from qimage import qimage
from qimage.img_transforms import Resize, ToMPS, ToTrivialMPS, NormalizeVector,\
    ToPatches, NormalizeMPS, FlattenPatches, NormalizePatches, RelativeNormMPS, ColorQubitMPS, Snake
from data_tracker import DataTracker 

def main(chi_tn, L,
        Nepochs=300, 
        batch_size=128,
        eval_size=1000,
        pd=(32,32),
        chi_img=4,
        **dataset_params):
    opt = optax.adam(1.e-4) 

    @jax.jit
    def update(tn,
               opt_state: optax.OptState,
               batch):
        grads = jax.grad(loss)(tn, batch)

        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(tn, updates)
        return tn, opt_state

    shape = (32,32)
    tn = init(L, chi_tn)

    train, test = qimage.get_dataset(**dataset_params)
    training_generator = DataLoader(train, batch_size=batch_size, collate_fn=qimage.numpy_collate)

    train_eval = DataLoader(train, batch_size=eval_size, collate_fn=qimage.numpy_collate)
    test_eval = DataLoader(train, batch_size=eval_size, collate_fn=qimage.numpy_collate)

    train_eval, test_eval = cycle(train_eval), cycle(test_eval)

    opt_state = opt.init(tn)
    attr = ["output", dataset_params['dataset_name'], f"size_{shape[0]}x{shape[1]}",\
            f"patch_{pd[0]}x{pd[1]}", f"chi_img{chi_img}"]

    dt = DataTracker(attr, experimental=False, overwrite=False, chi=chi_tn)

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
        for batch in training_generator:
            tn, opt_state = update(tn, opt_state, batch)
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
        dt.update(save_interval=1)

    dt.save()

def cluster_main(pd, chi_tn, chi_img):
    shape = (32,32)
    dataset_params = dict(transforms=[Resize(shape), ToPatches(pd),\
            FlattenPatches(), Snake(), ColorQubitMPS(chi_img)],\
            dataset_name="fashion-mnist")
    pixels_per_patch = np.prod(shape) // np.prod(pd)
    if pixels_per_patch == 1:
        L = np.prod(shape)
    else:
        L = int(np.prod(pd) * (int(np.ceil(np.log2(pixels_per_patch))) + 1))
    main(chi_tn, L,
        Nepochs=300, 
        batch_size=128,
        eval_size=1000,
        chi_img=chi_img,
        pd=pd,
        **dataset_params)

if __name__ == '__main__':
    cluster_main()
