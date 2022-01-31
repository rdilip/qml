"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import os

# $ -l h_vmem=4G,h_cpu=70:59:00,h_fsize=4G

config = {
    'jobname': 'qml',
    'email': 'olivergwayne@gmail.com',
    'sim_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_tn.py'),
    'sim_fct': 'cluster_main',
    'require': {
        'mem': '8G',
        'cpu': '23:59:00',
        'Nslots': 4,
        'filesize': '8G'
    },
    'params': []  # list of dictionaries (containing the kwargs given to the function `sim_fct`)
}

for pd in [(2, 1), (2, 2), (2, 4), (4,4)]:
    for chi_tn in [8, 10, 12, 14, 18, 20]:
        for chi_img in [2, 3, 4, 5, 6, 7, 8]:
            kwargs = dict(pd=pd, chi_tn=chi_tn, chi_img=chi_img, Nepochs=2000)
            config['params'].append(kwargs.copy())

cluster_jobs.submit_sge(config)    # our linux cluster at TUM Physics department
#  cluster_jobs.submit_slurm(config)  # NIM cluster at LRZ
#  cluster_jobs.run_local(config)     # alternative to run the simulation directly
