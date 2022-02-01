# coding: utf-8
""" Script to restart  jobs that got missed for some reason """
import subprocess
import re
import pickle

def get_restarts(config_file="qml.config.pkl"):
    with open(config_file, "rb") as f:
        config = pickle.load(f)
    params = config['params']

    bash_cmd = "qstat"
    process = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE)
    output, err = process.communicate()

    # -1 gives param indices because sge 1 indexes
    output
    active_jobs = [int(x[7:])-1 for x in re.findall('     4 \d+', str(output))]
    restart_dicts = []

    for i in range(len(params)):
        if i not in active_jobs:
            print(f"Missing {i}")
            restart_dicts.append(params[i])

    with open("restart_dicts.pkl", "wb+") as f:
        pickle.dump(restart_dicts, f)
    return restart_dicts
