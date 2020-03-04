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

job = 'job50'

base_dir = '/scratch/li.baol/tsrbrd_log/job_runs/archive/random_practical_finish_1/'
log_dir = base_dir + job + '/*'
dirs = glob.glob(log_dir)
dirs.sort()
pdb.set_trace()

loss_combine = []

for tc in dirs:
    iterator = EventAccumulator(tc).Reload()
    pdb.set_trace()
    tag = 'loss' #iterator.Tags()['scalars'][2] # this is tag for loss

    loss = [item.value for item in iterator.Scalars(tag)]
    loss_combine += loss
    wall_time = [t.wall_time for t in iterator.Scalars(tag)]
    pdb.set_trace()
    relative_time = [(time - wall_time[0])/3600 for time in wall_time]


