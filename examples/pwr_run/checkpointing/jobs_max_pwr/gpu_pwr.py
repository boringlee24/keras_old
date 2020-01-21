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

testcase = sys.argv[1] # job1
#print(testcase)
#dirs = glob.glob(log_dir)
#dirs.sort()
#pwr_all = []
#avg_all = []

base_dir = "/scratch/li.baol/GPU_pwr_meas/tensorflow/job_runs/"
file_path = base_dir + testcase + '.csv'

data = pandas.read_csv(file_path)
pwr = data[data.columns[2]].tolist()
pwr_array = np.asarray(pwr)
med_pwr = np.median(pwr_array)

# write the power value to power.json
power_dict = {}
with open('power.json', 'r') as fp:
    power_dict = json.load(fp)
power_dict[testcase] = med_pwr
with open('power.json', 'w') as fp:
    json.dump(power_dict, fp)


