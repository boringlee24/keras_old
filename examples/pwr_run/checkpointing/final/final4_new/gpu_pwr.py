import pandas
import numpy as np
import pdb

def process_csv(job_name, testcase):
    base_dir = "/scratch/li.baol/GPU_pwr_meas/tensorflow/job_runs/" + testcase + '/'
    file_path = base_dir + job_name + '.csv'
    
    data = pandas.read_csv(file_path)
    pwr = data[data.columns[2]].tolist()
    util = data[data.columns[5]].tolist()
    valid_pwr = []
    valid_util = []
    for i in range(len(util)):
        if util[i] > 0:
            valid_pwr.append(pwr[i])
            valid_util.append(util[i])
    
    pwr_array = np.asarray(valid_pwr)
    util_array = np.asarray(valid_util)
    med_pwr = np.median(pwr_array)
    med_util = np.median(util_array)
    return med_pwr, med_util

#a, b = process_csv('job6')
#pdb.set_trace()
#print()

