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

log_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/csv/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_time = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    time = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_time:
        all_time[model][gpu] = time
    else:
        all_time[model] = {gpu: time}

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("P100 time (h) vs K80 time (h) for 50 epochs")
NUM_COLORS = len(all_time)
cm = plt.get_cmap('tab20') #'gist_rainbow')
axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
for key in all_time:
    axs.scatter(all_time[key]['K80'], all_time[key]['P100'], label = key)

axs.set_xlabel('K80 time (h)')
axs.set_ylabel('P100 time (h)')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(5))
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs.grid(which='both', axis='both', linestyle=':', color='black')

plt.savefig("./time/P100_vs_K80.png")

# Now plot P100 power reduction(W) vs K80 power(W)

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("P100 time reduction(h) vs K80 time(h) for 50 epochs")
NUM_COLORS = len(all_time)
cm = plt.get_cmap('tab20') #'gist_rainbow')
axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
for key in all_time:
    axs.scatter(all_time[key]['K80'], all_time[key]['K80'] - all_time[key]['P100'], label = key)

axs.set_xlabel('K80 time (h)')
axs.set_ylabel('P100 time reduction (h)')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(5))
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs.grid(which='both', axis='both', linestyle=':', color='black')

plt.savefig("./time/P100_reduction_vs_K80.png")

# Now plot P100 power save ratio (%) vs K80 power(W)

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("P100 time save percentage (%) vs K80 time (h) for 50 epochs")
NUM_COLORS = len(all_time)
cm = plt.get_cmap('tab20') #'gist_rainbow')
axs.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
for key in all_time:
    axs.scatter(all_time[key]['K80'], (all_time[key]['K80'] - all_time[key]['P100']) / all_time[key]['K80'] * 100, label = key)

axs.set_xlabel('K80 time (W)')
axs.set_ylabel('P100 time save (%)')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(5))
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs.grid(which='both', axis='both', linestyle=':', color='black')

plt.savefig("./time/P100_save_vs_K80.png")

