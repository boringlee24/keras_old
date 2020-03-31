import os
import numpy as np
import pandas as pd
import sys
import pdb
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

testcase = sys.argv[1] # K80_vgg19_32
print(testcase)
base_dir = '/scratch/li.baol/tsrbrd_log/pwr_meas/round1/'
result_base_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/round1/'
log_dir = base_dir + testcase + '_*/'

dirs = glob.glob(log_dir)
dirs.sort()
time_all = [] # execution time for all 10 reps


for tc in dirs:
    model = tc.split('/')[5+1]
    iterator = EventAccumulator(tc).Reload()
    tag = iterator.Tags()['scalars'][2] # this is tag for loss

    wall_time = [t.wall_time for t in iterator.Scalars(tag)]
    relative_time = [(time - wall_time[0])/3600 for time in wall_time]
    time_all.append(relative_time[len(relative_time) - 1])

df = pd.DataFrame(time_all, columns=["time(h)"])
df.to_csv(result_base_dir + 'csv/' + testcase + '.csv', index=False)

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle(testcase +  " GPU time (h) to train 50 epochs")
    
x = np.arange(len(time_all))
axs.bar(x, time_all)
axs.set_xlabel('test rep (10 reps in total)')
axs.set_ylabel('time (h)')
#axs.set_yticks(minor=True)
axs.get_yaxis().set_minor_locator(MultipleLocator(5))

axs.grid(which='both', axis='y', linestyle=':', color='black')
time = int(sum(time_all) / len(time_all))
plt.savefig(result_base_dir + "png/" + testcase + '_time' + str(time) + ".png")
