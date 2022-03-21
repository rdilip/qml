""" This is a quick script I used to convert models from dictionary formats to 
multiple nump file formats.... because pickle likes to corrupt files
"""
import numpy as np
import os
import datetime
from pickle import UnpicklingError
from data_tracker import load
from tn_mps import *
from torch.utils.data import DataLoader

from qimage import qimage
from qimage.img_transforms import Resize, ToMPS, ToTrivialMPS, NormalizeVector,\
    ToPatches, NormalizeMPS, FlattenPatches, NormalizePatches, RelativeNormMPS, ColorQubitMPS, Snake
from itertools import cycle

with open("errors.txt", "a+") as f:
    f.write(str(datetime.datetime.now()) + "\n")

def get_fpath(pd, chi_img, chi_tn):
    return f"output/fashion-mnist/size_32x32/patch_{pd[0]}x{pd[1]}/chi_img{chi_img}/"

def log(msg):
    with open("errors.txt", "a+") as f:
        f.write(msg + "\n")

def reset_models(chi_imgs, chi_tns, pds):
    for chi_img in chi_imgs:
        for chi_tn in chi_tns:
            for pd in pds:
                print(chi_img, chi_tn, pd)
                fpath = get_fpath(pd, chi_img, chi_tn)
                if not os.path.exists(fpath + "OLD/"):
                    os.makedirs(fpath + "OLD/")
                if os.path.exists(fpath + "chi{chi_tn}_model.npy"):
                    raise ValueError
                try:
                    models = np.load(fpath + f"OLD/chi{chi_tn}_model.npy", allow_pickle=True)
                    for key in models[-1].keys():
                        data = np.array([m[key] for m in models])
                        np.save(fpath + f"chi{chi_tn}_model_{key}.npy", data)
                    # os.rename(fpath + f"chi{chi_tn}_model.npy", fpath + f"OLD/chi{chi_tn}_model.npy")
                except FileNotFoundError:
                    for key in ["left", "right", "center"]:
                        if os.path.exists(fpath + f"chi{chi_tn}_model_{key}.npy"):
                            os.remove(fpath + f"chi{chi_tn}_model_{key}.npy")
                except UnpicklingError as e:
                    log(f"{chi_img}, {chi_tn}, {pd}\t" + str(e))
                    for label in ["model", "loss", "time", "time_elapsed", "train_accuracy", "test_accuracy"]:
                        if os.path.exists(fpath + f"chi{chi_tn}_{label}.npy"):
                            os.rename(fpath + f"chi{chi_tn}_{label}.npy", fpath + f"OLD/chi{chi_tn}_{label}.npy")
                        for key in ["left", "right", "center"]:
                            if os.path.exists(fpath + f"chi{chi_tn}_model_{key}.npy"):
                                os.remove(fpath + f"chi{chi_tn}_model_{key}.npy")
                except OSError as e:
                    log(f"{chi_img}, {chi_tn}, {pd}\t" + str(e))
                    for label in ["model", "loss", "time", "time_elapsed", "train_accuracy", "test_accuracy"]:
                        if os.path.exists(fpath + f"chi{chi_tn}_{label}.npy"):
                            os.rename(fpath + f"chi{chi_tn}_{label}.npy", fpath + f"OLD/chi{chi_tn}_{label}.npy")
                        for key in ["left", "right", "center"]:
                            if os.path.exists(fpath + f"chi{chi_tn}_model_{key}.npy"):
                                os.remove(fpath + f"chi{chi_tn}_model_{key}.npy")
                       

def reset_accuracies(chi_imgs, chi_tns, pds):
    for chi_img in chi_imgs:
        for pd in pds:
            dataset_params = dict(transforms=[Resize((32,32)), ToPatches(pd), FlattenPatches(), Snake(),\
                    ColorQubitMPS(chi_img)], dataset_name="fashion-mnist")
            train, test = qimage.get_dataset(**dataset_params)
            train_eval = DataLoader(train, batch_size=1000, collate_fn=qimage.numpy_collate)
            test_eval = DataLoader(test, batch_size=1000, collate_fn=qimage.numpy_collate)
            train_eval, test_eval = cycle(train_eval), cycle(test_eval)
            for chi_tn in chi_tns:
                fpath = get_fpath(pd, chi_img, chi_tn)
                test_path = fpath + f"chi{chi_tn}_test_accuracy.npy"
                model = load(fpath + f"chi{chi_tn}_", "model", keys=["left", "center", "right"])

                data = []
                if os.path.exists(test_path):
                    data = np.load(test_path)
                if model is None:
                    continue # model removed, not recoverable
                if len(data) != len(model["left"]):
                    # Recover data by loading the models
                    print(chi_img, chi_tn, pd)
                    
                    new_train = np.zeros(shape=model["left"].shape[0])
                    new_test = np.zeros(shape=model["left"].shape[0])
                    new_loss = np.zeros(shape=model["left"].shape[0])
                    for i in range(1, 51):
                        tn = {}
                        for key in model:
                            tn[key] = model[key][-i]
                        new_test[-i] = accuracy(tn, next(test_eval))
                        new_train[-i] = accuracy(tn, next(train_eval))
                        new_loss[-i] = loss(tn, next(test_eval))
                    np.save(fpath + f"chi{chi_tn}_test_accuracy.npy", new_test)
                    np.save(fpath + f"chi{chi_tn}_train_accuracy.npy", new_train)
                    np.save(fpath + f"chi{chi_tn}_loss.npy", new_loss)
                            
if __name__ == '__main__':
    chi_imgs = [2, 3, 4, 5, 6, 7, 8]
    chi_tns = [8, 10, 12, 14, 16, 18, 20]
    pds = [(1,1), (2,1), (2,2), (2,4), (4,4)]
    # reset_models(chi_imgs, chi_tns, pds)
    reset_accuracies(chi_imgs, chi_tns, pds)

