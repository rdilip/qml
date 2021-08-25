""" DataTracker is a class to track and store temporal data, such as in neural
network training. Standard usage is

    >>> dt = DataTracker(attr, prepend=prepend)
    >>> dt.register("observation", obs_func())
    >>> for epoch in epochs:
    >>>     dt.update()

This will register an observation. The value of observation should be given by
obs_func(), which is usually just lambda: obs_value. Each call to dt.update()
will update and save the parameters.
"""

import os
import numpy as np
import pickle
import warnings
from math import log10, floor

class DataTracker:
    def __init__(self, attr, prepend="", experimental=False, overwrite=False):
        """ 
        Args:
            attr: list of attributes, each attribute generates a level in a 
                nested directory for saving.
            prepend: String to prepend onto saved files.
            experimental: bool. If True, does not save anything.
            overwrite: bool. If True, will overwrite existing data. If False,
                loads and appends to existing data
        """
            
        self.exp = experimental
        self.overwrite = overwrite
        if self.exp:
            warnings.warn("Experimental flag on: nothing will save!")
        self.fpath = "/".join(attr)
        if not os.path.exists(self.fpath):
            os.makedirs(self.fpath)
        if prepend:
            self.fpath += "/" + prepend + "_"


        self.tracked_params = {}
        self.param_data = {}
        self.Niter = 0

    def register(self, label, param_func):
        """
        Args:
            label: string, name of observation
            param_func: Callable, returns current parameter.
        """
        self.tracked_params[label] = param_func
        self.param_data[label] = [param_func()]
        if not self.overwrite:
            data = load(self.fpath, label)
            if not len(data):
                self.param_data[label] = [param_func()]
            else:
                self.param_data[label] = list(data)
        # some edge cases
        if label == "time_elapsed":
            self.tracked_params[label] = lambda: param_func() + self.param_data[label][0]
        return self.param_data[label]

    def update(self, save_interval=1):
        """
        Args:
            save_interval: Save every save_interval iterations.
        """
        if self.exp:
            return
        for label, param_func in self.tracked_params.items():
            self.param_data[label].append(param_func())
        self.Niter += 1
        if self.Niter % save_interval == 0:
            self.save()

    def save(self):
        """ Expanded save function that tries numpy, otherwise goes to pkl.
        """
        if self.exp:
            return
        for label, data in self.param_data.items():
            try:
                np.save(self.fpath + label + ".npy", data, allow_pickle=True)
            except:
                with open(self.fpath + label + ".pkl", "wb+") as f:
                    pickle.dump(data, f)

def load(fpath, label):
    """ Expanded load function that tries numpy, otherwise goes to pkl.
    """
    try:
        data = np.load(fpath + label + ".npy", allow_pickle=True)
    except FileNotFoundError:
        try:
            with open(fpath + label + ".pkl", "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = []
    return data

def round_sf(x, m=1):
    """ Round to m significant figures """
    return round(x, -int(floor(log10(abs(x)))) + m - 1)

