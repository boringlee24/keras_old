import pandas
import pdb
from datetime import datetime
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import sys
from matplotlib.ticker import MultipleLocator
import json

log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/job_runs/'
dirs = glob.glob(log_dir)
dirs.sort()
pwr_all = []
avg_all = []

pwr_dict = {}
util_dict = {}

for i in range(50):
    job_name = 'job' + str(i+1) + '.csv'
    file_path = log_dir + job_name
    data = pandas.read_csv(file_path)
    pwr = data[data.columns[2]].tolist()
    util = data[data.columns[5]].tolist()

    valid_pwr = []
    valid_util = []
    for j in range(len(pwr)):
        if util[j] > 0:
            valid_pwr.append(pwr[j])
            valid_util.append(util[j])
    pwr_array = np.asarray(valid_pwr)
    util_array = np.asarray(valid_util)
    if (len(valid_pwr) == 0):
        pwr_dict[str(i+1)] = 0
        util_dict[str(i+1)] = 0
    else:
        pwr_dict[str(i+1)] = np.median(pwr_array)
        util_dict[str(i+1)] = np.median(util_array)

with open('pwr.json', 'w') as fp1:
    json.dump(pwr_dict, fp1, sort_keys=False, indent=4)
with open('util.json', 'w') as fp1:
    json.dump(util_dict, fp1, sort_keys=False, indent=4)

