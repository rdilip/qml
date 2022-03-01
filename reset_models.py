""" This is a quick script I used to convert models from dictionary formats to 
multiple nump file formats.... because pickle likes to corrupt files
"""
import numpy as np
import os
import datetime
from pickle import UnpicklingError

with open("errors.txt", "a+") as f:
    f.write(str(datetime.datetime.now()) + "\n")

def get_fpath(pd, chi_img, chi_tn):
    return f"output/fashion-mnist/size_32x32/patch_{pd[0]}x{pd[1]}/chi_img{chi_img}/"

def log(msg):
    with open("errors.txt", "a+") as f:
        f.write(msg + "\n")

if __name__ == '__main__':
    chi_imgs = [2, 3, 4, 5, 6, 7, 8]
    chi_tns = [8, 10, 12, 14, 16, 18, 20]
    pds = [(1,1), (2,1), (2,2), (2,4), (4,4)]
    for chi_img in chi_imgs:
        for chi_tn in chi_tns:
            for pd in pds:
                print(chi_img, chi_tn, pd)
                fpath = get_fpath(pd, chi_img, chi_tn)
                if not os.path.exists(fpath + "OLD/"):
                    os.makedirs(fpath + "OLD/")
                try:
                    models = np.load(fpath + f"chi{chi_tn}_model.npy", allow_pickle=True)
                    for key in models[0].keys():
                        data = np.array([m[key] for m in models])
                        np.save(fpath + f"chi{chi_tn}_model_{key}.npy", data)
                    os.rename(fpath + f"chi{chi_tn}_model.npy", fpath + f"OLD/chi{chi_tn}_model.npy")
                except FileNotFoundError:
                    continue
                except UnpicklingError as e:
                    log(f"{chi_img}, {chi_tn}, {pd}\t" + str(e))
                    for label in ["model", "loss", "time", "time_elapsed", "train_accuracy", "test_accuracy"]:
                        if os.path.exists(fpath + f"chi{chi_tn}_{label}.npy"):
                            os.rename(fpath + f"chi{chi_tn}_{label}.npy", fpath + f"OLD/chi{chi_tn}_{label}.npy")
                except OSError as e:
                    log(f"{chi_img}, {chi_tn}, {pd}\t" + str(e))
                    for label in ["model", "loss", "time", "time_elapsed", "train_accuracy", "test_accuracy"]:
                        if os.path.exists(fpath + f"chi{chi_tn}_{label}.npy"):
                            os.rename(fpath + f"chi{chi_tn}_{label}.npy", fpath + f"OLD/chi{chi_tn}_{label}.npy")

 
