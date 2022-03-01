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
import time
from math import log10, floor

class DataTracker:
    def __init__(self, attr, experimental=False, overwrite=False, **kwargs):
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
        print("Saving to: " + self.fpath)
        if len(kwargs) > 0:
            for label in kwargs:
                self.fpath += "/" + f"{label}{kwargs[label]}" + "_"

        
        self.start_time = time.time()
        self.param_data = {"time": [self.start_time]}
        self.tracked_params = {"time": lambda: time.time() - self.start_time}
        self.save_intervals = {"time": 1}
        self.Niter = 0

    def register(self, label, param_func, save_interval=1, is_dict=False):
        """
        Args:
            label: string, name of observation
            param_func: Callable, returns current parameter.
        """
        # TODO make this nicer
        initial_data = param_func()

        if is_dict:
            for key in initial_data:
                self.save_intervals[label + f"_{key}"] = save_interval
                self.param_data[label + f"_{key}"] = [initial_data[key]]
                self.tracked_params[label + f"_{key}"] = lambda: param_func()[key]
        else:
            self.save_intervals[label] = save_interval
            self.param_data[label] = [initial_data]
            self.tracked_params[label] = param_func

        if not self.overwrite:
            if is_dict:
                keys = initial_data.keys()
                data = load(self.fpath, label, keys=keys)
                if data is not None:
                    for key in keys:
                        self.param_data[label + f"_{key}"] = list(data[key])
            else:
                data = load(self.fpath, label)
                if data is not None:
                    self.param_data[label] = list(data)

        # For dicts, e.g., a model, return just the last element. For all other cases,
        # return the full list.
        if is_dict:
            output = {}
            for key in keys:
                output[key] = self.param_data[label + f"_{key}"][-1]
        else:
            output = self.param_data[label]
        return output

    def update(self):
        """
        Args:
            save_interval: Save every save_interval iterations.
        """
        if self.exp:
            return
        for label, param_func in self.tracked_params.items():
            self.param_data[label].append(param_func())
        self.Niter += 1
        for param in self.tracked_params:
            if self.Niter % self.save_intervals[param] == 0:
                self.save(param)

    def save(self, param_label):
        if self.exp:
            return
        data = self.param_data[param_label]
        np.save(self.fpath + param_label + ".npy", data, allow_pickle=True)

    def save_all(self):
        for param in self.tracked_params:
            self.save(param)

def load(fpath, label, keys=None):
    """ Expanded load function that tries numpy, otherwise goes to pkl.
    """
    try:
        if keys is None:
            data = np.load(fpath + label + ".npy", allow_pickle=False)
        else:
            data = {}
            for key in keys:
                data[key] = np.load(fpath + label + f"_{key}.npy", allow_pickle=False)
    except FileNotFoundError:
        data = None
    return data

    
