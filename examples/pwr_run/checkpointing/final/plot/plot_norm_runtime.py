import os
import numpy as np
import pandas as pd
import sys
import pdb
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas

testcases = ['baseline', 'baseline_plus', 'scheme', 'feedback', 'no_threshold', 'no_safeguard', 'start_on_both',
'predict_error']

testcases1 = ['baseline', 'baseline_plus', 'feedback', 'scheme', 'start_on_both']

normed_dir = '/home/li.baol/GIT/keras_old/examples/pwr_run/checkpointing/final/comparison/norm_JCT/'
fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("normalized runtime (to baseline)")


for test in testcases1:
    file_path = normed_dir + test + '_jct.csv'
    data = pandas.read_csv(file_path, header=None)
    normed_jct = data[data.columns[0]].tolist()
    sort = sorted(normed_jct)

    axs.plot(sort, label = test)

axs.set_xlabel('jobs (sorted)') #'training time (h)')
axs.set_ylabel('reference JCT / scheme JCT')
axs.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
axs.grid(which='both', axis='both', linestyle=':', color='black')

#plt.show()
plt.savefig('normalized_runtime_to_baseline.png')



'''
testcases = {
'configuration of resnet50 on K80': 'K80_resnet50_32', 
'configuration of resnet50 on P100': 'P100_resnet50_32',
'configuration of densenet201 on K80': 'K80_densenet201_32',
'configuration of densenet201 on P100': 'P100_densenet201_32' }

base_dir = '/scratch/li.baol/tsrbrd_log/pwr_meas/round1/'

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("accuracy curve during training")

for key, value in testcases.items():
    log_dir = base_dir + value + '_*/'
    
    dirs = glob.glob(log_dir)
    dirs.sort()
    time_all = [] # time for all 10 reps
    accuracy_all = []
    
    for tc in dirs:
        model = tc.split('/')[5+1]
        iterator = EventAccumulator(tc).Reload()
        tag = iterator.Tags()['scalars'][3] # this is tag for accuracy
    
        accuracy = [item.value for item in iterator.Scalars(tag)]
        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
        relative_time = [(time - wall_time[0])/3600 for time in wall_time]
    
        time_all.append(relative_time)
        accuracy_all.append(accuracy)
    
    time_avg = [0] * len(time_all[0])
    accuracy_avg = [0] * len(time_all[0])
    
    for j in range(len(time_all)): # 10
        time_avg = np.add(time_all[j], time_avg)
        accuracy_avg = np.add(accuracy_all[j], accuracy_avg) 
    
    time_avg = time_avg / len(time_all)
    accuracy_avg = accuracy_avg / len(time_all) * 100
    
    clr = 'tab:orange' if 'resnet50' in value else 'tab:blue'
    mkr = 'o' if 'K80' in value else '^'
    axs.plot(#time_avg, 
accuracy_avg, label = key, color = clr, marker = mkr)

axs.set_xlabel('epoch') #'training time (h)')
axs.set_ylabel('model accuracy (%)')

axs.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
axs.get_xaxis().set_minor_locator(MultipleLocator(0.5))
axs.get_yaxis().set_minor_locator(MultipleLocator(5))
axs.grid(which='both', axis='both', linestyle=':', color='black')

plt.savefig('accuracy_curve_insufficient_epoch.png')

#pdb.set_trace()
#df = pd.DataFrame(time_all, columns=["time(h)"])
#df.to_csv(result_base_dir + 'csv/' + testcase + '.csv', index=False)
#
#fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
#fig.suptitle(testcase +  " GPU time (h) to train 50 epochs")
#    
#x = np.arange(len(time_all))
#axs.bar(x, time_all)
#axs.set_xlabel('test rep (10 reps in total)')
#axs.set_ylabel('time (h)')
##axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(5))
#
#axs.grid(which='both', axis='y', linestyle=':', color='black')
#time = int(sum(time_all) / len(time_all))
#plt.savefig(result_base_dir + "png/" + testcase + '_time' + str(time) + ".png")
'''
