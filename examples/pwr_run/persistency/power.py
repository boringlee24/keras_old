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

testcase = 'K80_mobilenet_32_1' #sys.argv[1] # K80_mobilenet_32
print(testcase)
base_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/'
log_dir = base_dir + testcase + '/sample*.csv' # /scratch/li.baol/GPU_pwr_meas/pytorch/K80_mobilenet_32_*/
dirs = glob.glob(log_dir)
dirs.sort()
pwr_all = []
avg_all = []

counter = 0
median_pwr = []
pwr = []

for i in range(len(dirs)):
    file_path = base_dir + testcase + '/sample_' + str(i) + '.csv'
    counter += 1
    try: # in case the file is empty
        data = pandas.read_csv(file_path)
        pwr = data[data.columns[3]].tolist() + pwr
        if counter == 60:
            counter = 0
            pwr_array = np.asarray(pwr)
            pwr = []
            median_pwr.append(np.median(pwr_array))
    except pandas.errors.EmptyDataError:
        print('error: empty data')

mean = np.mean(median_pwr)
stdev = np.std(median_pwr)
vc_pct = stdev/mean * 100
print(median_pwr)
print('mean is ' + str(mean))
print('standard deviation is ' + str(stdev))
print('variation coefficient is ' + str(vc_pct) + '%')

