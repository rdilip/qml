import os
import numpy as np
import pickle

class DataTracker:
    def __init__(self, attr, prepend=""):
        self.fpath = "/".join(attr)
        if not os.path.exists(self.fpath):
            os.makedirs(self.fpath)
        if prepend:
            self.fpath += "/" + prepend + "_"


        self.tracked_params = {}
        self.param_data = {}
        self.Niter = 0

    def register(self, label, param_func):
        self.tracked_params[label] = param_func
        self.param_data[label] = [param_func()]

    def update(self, save_interval=1):
        for label, param_func in self.tracked_params.items():
            self.param_data[label].append(param_func())
        self.Niter += 1
        if self.Niter % save_interval == 0:
            self.save()

    def save(self):
        for label, data in self.param_data.items():
            try:
                np.save(self.fpath + label + ".npy", data)
            except:
                with open(self.fpath + label + ".pkl", "wb+") as f:
                    pickle.dump(data, f)
