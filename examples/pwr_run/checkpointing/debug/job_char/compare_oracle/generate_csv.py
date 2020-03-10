import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)
with open('oracle_JCT.json', 'r') as fp:
    oracle_only = json.load(fp)
with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)
with open('oracle_K80_time.json', 'r') as fp:
    oracle_K80_only = json.load(fp)
with open('oracle_V100_time.json', 'r') as fp:
    oracle_V100_only = json.load(fp)
with open('oracle_overhead.json', 'r') as fp:
    oracle_overhead_only = json.load(fp)
with open('oracle_epoch_waste.json', 'r') as fp:
    oracle_epoch_waste_only = json.load(fp)
with open('oracle_num_mig.json', 'r') as fp:
    oracle_num_mig_only = json.load(fp)
with open('total_time.json', 'r') as fp:
    total_time_only = json.load(fp)
with open('load_time.json', 'r') as fp:
    load_time_only = json.load(fp)
with open('save_time.json', 'r') as fp:
    save_time_only = json.load(fp)
with open('speedup.json', 'r') as fp:
    speedup_only = json.load(fp)

job_list = []
oracle = []
k80 = []
v100 = []
oracle_K80 = []
oracle_V100 = []
oracle_overhead = []
oracle_epoch_waste = []

oracle_num_mig = []
total_time = []
load_time = []
save_time = []
speedup = []

for i in range(50):
    job = str(i+1)
    job_list.append('job'+job)
    oracle.append(oracle_only[job])
    k80.append(k80_only[job])
    v100.append(v100_only[job])
    oracle_K80.append(oracle_K80_only[job])
    oracle_V100.append(oracle_V100_only[job])
    oracle_overhead.append(oracle_overhead_only[job])
    oracle_epoch_waste.append(oracle_epoch_waste_only['job'+job])
    oracle_num_mig.append(oracle_num_mig_only[job])
    total_time.append(total_time_only[job])
    load_time.append(load_time_only[job])
    save_time.append(save_time_only[job])
    speedup.append(round(speedup_only[job],2))

job_list = np.asarray(job_list)
oracle = np.asarray(oracle)
k80 = np.asarray(k80)
v100 = np.asarray(v100)
oracle_K80 = np.asarray(oracle_K80)
oracle_V100 = np.asarray(oracle_V100)
oracle_overhead = np.asarray(oracle_overhead)
oracle_epoch_waste = np.asarray(oracle_epoch_waste)

oracle_num_mig = np.asarray(oracle_num_mig)
total_time = np.asarray(total_time)
load_time = np.asarray(load_time)
save_time = np.asarray(save_time)
speedup = np.asarray(speedup)

rows = zip(job_list, k80, v100, oracle, oracle_K80, oracle_V100, oracle_overhead, oracle_epoch_waste, oracle_num_mig,
total_time, save_time, load_time, speedup)
with open('comparison.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

#np.savetxt('comparison.csv', (job_list, k80, v100, oracle, oracle_K80, oracle_V100, oracle_overhead, oracle_num_mig,
#total_time, save_time, load_time), fmt='%s')

