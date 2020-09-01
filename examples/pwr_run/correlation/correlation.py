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
from scipy.stats import pearsonr, spearmanr

log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/util_csv/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_pwr = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6+1].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    pwr = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_pwr:
        all_pwr[model][gpu] = pwr
    else:
        all_pwr[model] = {gpu: pwr}

log_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/round1/csv/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_time = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6+1].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    time = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_time:
        all_time[model][gpu] = time
    else:
        all_time[model] = {gpu: time}

# Now plot P100 power save ratio (%) vs K80 power(W)

x_data = []
y_data = []

for key in all_pwr:
    if ('mnasnet' not in key and 'mobilenet' not in key):
        K80_energy = all_pwr[key]['K80'] * all_time[key]['K80'] / 1000
        P100_energy = all_pwr[key]['P100'] * all_time[key]['P100'] / 1000
        for i in all_pwr[key]['K80'].tolist(): # power
#        for i in (50 / all_time[key]['K80']).tolist(): # speed
#        for i in all_time[key]['K80'].tolist(): # time
            x_data.append(i)
#        for i in ((K80_energy - P100_energy) / K80_energy * 100).tolist(): # energy save
        for i in ((all_time[key]['K80'] - all_time[key]['P100']) / all_time[key]['K80'] * 100).tolist(): # speed up  
            y_data.append(i)

pcorr, pp = pearsonr(x_data, y_data)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))

scorr, sp = spearmanr(x_data, y_data)
print('Spearmans correlation: %.3f, p value %.3f' % (scorr, sp))



