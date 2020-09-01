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

model_list = ['densenet121', 'densenet169', 'densenet201', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19']
param_list = [6964106, 12501130, 18112138, 23555082, 42573322, 58240010, 14790666, 20100362]
flop_list = [14179748, 25478292, 36912436, 47216657, 85357840, 116783631, 29579613, 40196448]

log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/csv/*'
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

x_param = []
x_flop = []
y_save = []
y_speedup = []

for i in range(len(model_list)):
    key = model_list[i] + '_32'
    K80_energy = np.mean(all_pwr[key]['K80']) * np.mean(all_time[key]['K80']) / 1000
    P100_energy = np.mean(all_pwr[key]['P100']) * np.mean(all_time[key]['P100']) / 1000
    
    # correlation with network parameter
    x_param.append(param_list[i])
    x_flop.append(flop_list[i])
    y_save.append((K80_energy - P100_energy) / K80_energy * 100)
    y_speedup.append((np.mean(all_time[key]['K80']) - np.mean(all_time[key]['P100'])) / np.mean(all_time[key]['K80']) * 100)

print('energy save with parameter')
pcorr, pp = pearsonr(x_param, y_save)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
scorr, sp = spearmanr(x_param, y_save)
print('Spearmans correlation: %.3f, p value %.3f' % (scorr, sp))

print('energy save with FLOPs')
pcorr, pp = pearsonr(x_flop, y_save)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
scorr, sp = spearmanr(x_flop, y_save)
print('Spearmans correlation: %.3f, p value %.3f' % (scorr, sp))

print('speed up with parameter')
pcorr, pp = pearsonr(x_param, y_speedup)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
scorr, sp = spearmanr(x_param, y_speedup)
print('Spearmans correlation: %.3f, p value %.3f' % (scorr, sp))

print('speed up with FLOPs')
pcorr, pp = pearsonr(x_flop, y_speedup)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
scorr, sp = spearmanr(x_flop, y_speedup)
print('Spearmans correlation: %.3f, p value %.3f' % (scorr, sp))


