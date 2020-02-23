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
import os

job_name = sys.argv[1] # job1
#print(job_name)
#dirs = glob.glob(log_dir)
#dirs.sort()
#pwr_all = []
#avg_all = []

base_dir = "/scratch/li.baol/GPU_pwr_meas/tensorflow/job_runs/"
file_path = base_dir + job_name + '.csv'

data = pandas.read_csv(file_path)
pwr = data[data.columns[2]].tolist()
pwr_array = np.asarray(pwr)
med_pwr = np.median(pwr_array)

# write the power value to power.json
power_dict = {}
while True:
    if os.path.exists('power.json'):
        os.rename('power.json', 'power_lock.json')
        break
    else:
        time.sleep(1)
with open('power_lock.json', 'r') as fp:
    power_dict = json.load(fp)
power_dict[job_name] = med_pwr
json_file2 = json.dumps(power_dict)
with open('power_lock.json', 'w') as fp:
    fp.write(json_file2)
os.rename('power_lock.json', 'power.json')


