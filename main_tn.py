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
from qimage.img_transforms import Channel, ToMPS, Resize, ToTrivialMPS
from data_tracker import DataTracker 

def main(chi_tn, 
        Nepochs=300, 
        dataset="mnist",
        batch_size=128,
        eval_size=1000,
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

    L = np.prod(shape)
    tn = init(L, chi_tn)

    train, test = qimage.get_dataset(**dataset_params)
    training_generator = DataLoader(train, batch_size=batch_size, collate_fn=qimage.numpy_collate)

    train_eval = DataLoader(train, batch_size=eval_size, collate_fn=qimage.numpy_collate)
    test_eval = DataLoader(train, batch_size=eval_size, collate_fn=qimage.numpy_collate)

    train_eval, test_eval = cycle(train_eval), cycle(test_eval)

    opt_state = opt.init(tn)
    #attr = ["raw", "cpu", dataset, f"size_{shape[0]}x{shape[1]}", kernel,\
    #        f"patch_{pd[0]}x{pd[1]}", f"chi_img{chi_img}"]
    attr = "test"
    prepend = f"chi{chi_tn}"

    dt = DataTracker(attr, prepend=prepend, experimental=False, overwrite=True)

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

if __name__ == '__main__':
    chi_tn = 16
    chi_img = 1
    dataset_params = dict(
                        transforms=[Resize((32,32)), Channel("diff"), ToTrivialMPS()],\
                        transform_labels=["resize_32x32", "channel_diff", "trivial_mps"]
                        )
    main(chi_tn,
        Nepochs=300, 
        dataset="fashion-mnist",
        batch_size=128,
        eval_size=1000,
        **dataset_params)


