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

testcase = 'K80_mobilenet_32_1' 
base_dir = '/scratch/li.baol/tsrbrd_log/pwr_meas/round1/' 
log_dir = base_dir + testcase + '/*'
dirs = glob.glob(log_dir)
dirs.sort()
tc = dirs[0]

iterator = EventAccumulator(tc).Reload()
tag = 'loss'
wall_time = [t.wall_time for t in iterator.Scalars(tag)]
training_time = []

for i in range(len(wall_time)):
    if i > 0:
        training_time.append((wall_time[i] - wall_time[i-1]))
    
mean = np.mean(training_time)
stdev = np.std(training_time)
vc_pct = stdev/mean * 100
print(training_time)
print('mean is ' + str(mean))
print('standard deviation is ' + str(stdev))
print('variation coefficient is ' + str(vc_pct) + '%')


