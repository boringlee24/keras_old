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

#log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/regression/pwr/*'
log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/regression/util/*'

dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_pwr = {} # {densenet121_32:{P100:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[8].split('.')[0]
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
all_time = {} # {densenet121_32:{P100:a, K100:b}...}

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

# plot P100 time vs P100 power

###fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
###fig.suptitle("P100 time(h) vs P100 power(W)")
###NUM_COLORS = len(all_pwr)
###cm = plt.get_cmap('tab20') #'gist_rainbow')
###axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
###for key in all_pwr:
###    if ('mnasnet' not in key and 'mobilenet' not in key):
###        axs.scatter(all_pwr[key]['P100'], all_time[key]['P100'], label = key)
###
###axs.set_xlabel('P100 power (W)')
###axs.set_ylabel('P100 time (h)')
####axs.set_yticks(minor=True)
####axs.get_yaxis().set_minor_locator(MultipleLocator(5))
###box = axs.get_position()
###axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
###
###axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
###
###axs.grid(which='both', axis='both', linestyle=':', color='black')
###
###plt.savefig("./time_vs_power/P100_time_vs_P100.png")
###
#### plot V100 time vs P100 power
###
###fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
###fig.suptitle("V100 time(h) vs P100 power(W)")
###NUM_COLORS = len(all_pwr)
###cm = plt.get_cmap('tab20') #'gist_rainbow')
###axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
###for key in all_pwr:
###    if ('mnasnet' not in key and 'mobilenet' not in key):
###        axs.scatter(all_pwr[key]['P100'], all_time[key]['V100'], label = key)
###
###axs.set_xlabel('P100 power (W)')
###axs.set_ylabel('V100 time (h)')
####axs.set_yticks(minor=True)
####axs.get_yaxis().set_minor_locator(MultipleLocator(5))
###box = axs.get_position()
###axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
###
###axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
###
###axs.grid(which='both', axis='both', linestyle=':', color='black')
###
###plt.savefig("./time_vs_power/V100_time_vs_P100.png")
###
#### Now plot V100 power reduction(W) vs P100 power(W)
###
###fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
###fig.suptitle("V100 time reduction(h) vs P100 power(W)")
###NUM_COLORS = len(all_pwr)
###cm = plt.get_cmap('tab20') #'gist_rainbow')
###axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
###for key in all_pwr:
###    if ('mnasnet' not in key and 'mobilenet' not in key):
###        axs.scatter(all_pwr[key]['P100'], all_time[key]['P100'] - all_time[key]['V100'], label = key)
###
###axs.set_xlabel('P100 power (W)')
###axs.set_ylabel('V100 time reduction (h)')
####axs.set_yticks(minor=True)
####axs.get_yaxis().set_minor_locator(MultipleLocator(5))
###box = axs.get_position()
###axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
###
###axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
###
###axs.grid(which='both', axis='both', linestyle=':', color='black')
###
###plt.savefig("./time_vs_power/V100_reduction_vs_P100.png")

# Now plot V100 power save ratio (%) vs V100 power(W)

x_data = []
y1_data = []
y2_data = []
fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,10))
fig.suptitle("V100 time save percentage (%) vs V100 util(%)")
NUM_COLORS = len(all_pwr)
cm = plt.get_cmap('tab20') #'gist_rainbow')
axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
for key in all_pwr:
    pdb.set_trace()
#    if ('mnasnet' not in key and 'mobilenet' not in key):
    axs.scatter(all_pwr[key]['V100'], (all_time[key]['K80'] - all_time[key]['V100']) / all_time[key]['K80'] * 100, label = key)
    x_data += all_pwr[key]['V100'].tolist()
    y1_data += ((all_time[key]['K80'] - all_time[key]['V100']) / all_time[key]['K80'] * 100).tolist()
    y2_data += (all_time[key]['K80'] / all_time[key]['V100']).tolist()

with open('x_data.json', 'w') as outfile:
    json.dump(x_data, outfile)
with open('y1_data.json', 'w') as outfile:
    json.dump(y1_data, outfile)
with open('y2_data.json', 'w') as outfile:
    json.dump(y2_data, outfile)

axs.set_xlabel('V100 util (%)')
axs.set_ylabel('V100 time save (%)')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(5))
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs.grid(which='both', axis='both', linestyle=':', color='black')

#plt.show()
plt.savefig("./time_vs_util/V100_save_vs_V100.png")

