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
import random

##testcases = {
##'configuration of resnet50 on K80': 'K80_resnet50_32', 
##'configuration of resnet50 on P100': 'P100_resnet50_32',
##'configuration of densenet201 on K80': 'K80_densenet201_32',
##'configuration of densenet201 on P100': 'P100_densenet201_32' }

param_dict = {'densenet121': 6964106, 'densenet169': 12501130, 'densenet201': 18112138, 
              'resnet50': 23555082, 'resnet101': 42573322, 'resnet152': 58240010, 
              'vgg16': 14790666, 'vgg19': 20100362, 
              'mnasnet': 4243548, 'mobilenet': 2236682
              }

param_dict_sorted = {k: v for k, v in sorted(param_dict.items(), key=lambda item: item[1], reverse=True)}

base_dir = '/scratch/li.baol/tsrbrd_log/pwr_meas/round3/'
log_dir = base_dir + 'K80_*'
dirs = glob.glob(log_dir)
dirs.sort()

##fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
##fig.suptitle("accuracy curve during training")

time_all = [] # time for all 10 reps
accuracy_all = []

killed_job = []
promoted_job = []
unpromoted_job = []
exe_time = []
P_time = []

promote_prob = 40

promote_scheme = 'random'
unkilled_jobs = 36
promoted_jobs = 11
promote_list = random.sample(range(unkilled_jobs), promoted_jobs)

for tc in dirs:
    index = dirs.index(tc)
    #model = tc.split('/')[5+1]
    iterator_K = EventAccumulator(tc).Reload()
    tag_K = iterator_K.Tags()['scalars'][3] # this is tag for accuracy

    accuracy_K = [item.value for item in iterator_K.Scalars(tag_K)]
    wall_time_K = [t.wall_time for t in iterator_K.Scalars(tag_K)]
    relative_time_K = [(time - wall_time_K[0])/3600 for time in wall_time_K]

    tc = tc.replace('K80', 'P100')
    iterator_P = EventAccumulator(tc).Reload()
    tag_P = iterator_P.Tags()['scalars'][3] # this is tag for accuracy

    accuracy_P = [item.value for item in iterator_P.Scalars(tag_P)]
    wall_time_P = [t.wall_time for t in iterator_P.Scalars(tag_P)]
    relative_time_P = [(time - wall_time_P[0])/3600 for time in wall_time_P]

    # early termination condition: no accuracy > 0.5 for 1st 5 epochs
    if (max(accuracy_K[0:5]) <= 0.5):
        killed_job.append(tc)
        exe_time.append(relative_time_K[4])
    else: # execute this job till the end
        if promote_scheme == 'random':
            if np.random.uniform(0, 100) < promote_prob:
                promoted_job.append(tc)
                # promote job in this case
                exe_time1 = (relative_time_K[4]) # time spent before promotion
                exe_time2 = relative_time_P[len(relative_time_P) - 1] - relative_time_P[4]
                exe_time.append(exe_time1 + exe_time2)
                P_time.append(exe_time2)
            else:
                unpromoted_job.append(tc)
                # job not promoted in this case
                exe_time.append(relative_time_K[len(relative_time_K) - 1])
        elif promote_scheme == 'param':
            cnt = 0


total_exe_time = sum(exe_time)
avg_exe_time = np.mean(exe_time)
print('total time is ' + str(total_exe_time) + 'h, average time is ' + str(avg_exe_time) + 'h')
print('P100 execution time is ' + str(sum(P_time)) + 'h')
print('promoted jobs: ' + str(len(promoted_job)) + ', unpromoted jobs: ' + str(len(unpromoted_job)))



##time_avg = [0] * len(time_all[0])
##accuracy_avg = [0] * len(time_all[0])
##
##for j in range(len(time_all)): # 10
##    time_avg = np.add(time_all[j], time_avg)
##    accuracy_avg = np.add(accuracy_all[j], accuracy_avg) 
##
##time_avg = time_avg / len(time_all)
##accuracy_avg = accuracy_avg / len(time_all) * 100
##
##clr = 'tab:orange' if 'resnet50' in value else 'tab:blue'
##mkr = 'o' if 'K80' in value else '^'
##axs.plot(#time_avg, 
##accuracy_avg, label = key, color = clr, marker = mkr)
##
##axs.set_xlabel('epoch') #'training time (h)')
##axs.set_ylabel('model accuracy (%)')
##
##axs.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
##axs.get_xaxis().set_minor_locator(MultipleLocator(0.5))
##axs.get_yaxis().set_minor_locator(MultipleLocator(5))
##axs.grid(which='both', axis='both', linestyle=':', color='black')
##
##plt.savefig('accuracy_curve_insufficient_epoch.png')

