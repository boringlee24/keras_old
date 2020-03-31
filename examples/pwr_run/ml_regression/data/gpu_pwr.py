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

testcase = sys.argv[1] # K80_vgg19_32
print(testcase)
base_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/'
log_dir = base_dir + testcase + '_*/' # /scratch/li.baol/GPU_pwr_meas/pytorch/K80_vgg19_32_*/
dirs = glob.glob(log_dir)
dirs.sort()
pwr_all = []
avg_all = []

for tc in dirs: # try number
    model = tc.split('/')[5+1]
    files = glob.glob(tc + "sample*.csv")
    files.sort()
    avg_pwr = [0] * (len(files) + 1)
    
    for fil in files: # minute number
        file_path = fil
        minute = int(fil.split('/')[6+1].split('_')[1].split('.')[0])
        try: # in case the file is empty
            data = pandas.read_csv(file_path)
            pwr = data[data.columns[2]].tolist()
            util = data[data.columns[5]].tolist()
            valid = []
            for i in range(len(pwr)):
                if util[i] > 0:
                    valid.append(pwr[i])
            pwr_array = np.asarray(valid)
            if (len(valid) == 0):
                avg_pwr[minute] = 0
            else:
                avg_pwr[minute] = np.average(pwr_array)
        except pandas.errors.EmptyDataError:
            avg_pwr[minute] = 0
            pass
    pwr_all.append(avg_pwr)
    avg_pwr_filter = [i for i in avg_pwr if i > 10] # remove power measurements below 10W
    avg_all.append(np.median(avg_pwr_filter))
    
df = pandas.DataFrame(avg_all, columns=["power(W)"])
df.to_csv(base_dir + 'regression/pwr/' + testcase + '.csv', index=False)
