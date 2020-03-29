import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)
with open('final1_inverse_JCT.json', 'r') as fp:
    final1_inverse_only = json.load(fp)
with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)
with open('final1_inverse_K80_time.json', 'r') as fp:
    final1_inverse_K80_only = json.load(fp)
with open('final1_inverse_V100_time.json', 'r') as fp:
    final1_inverse_V100_only = json.load(fp)
with open('final1_inverse_overhead.json', 'r') as fp:
    final1_inverse_overhead_only = json.load(fp)
with open('final1_inverse_epoch_waste.json', 'r') as fp:
    final1_inverse_epoch_waste_only = json.load(fp)
with open('final1_inverse_num_mig.json', 'r') as fp:
    final1_inverse_num_mig_only = json.load(fp)
with open('final1_inverse_ovhd_a.json', 'r') as fp:
    final1_inverse_ovhd_a_only = json.load(fp)
with open('final1_inverse_ovhd_b.json', 'r') as fp:
    final1_inverse_ovhd_b_only = json.load(fp)
with open('final1_inverse_ovhd_c.json', 'r') as fp:
    final1_inverse_ovhd_c_only = json.load(fp)
with open('final1_inverse_ovhd_d.json', 'r') as fp:
    final1_inverse_ovhd_d_only = json.load(fp)
with open('final1_inverse_k80_1st.json', 'r') as fp:
    final1_inverse_k80_1st_only = json.load(fp)
with open('final1_inverse_v100_1st.json', 'r') as fp:
    final1_inverse_v100_1st_only = json.load(fp)
with open('speedup.json', 'r') as fp:
    speedup_only = json.load(fp)
with open('epoch_num.json', 'r') as fp:
    epoch_num_only = json.load(fp)
with open('k80_time.json', 'r') as fp:
    k80_time_only = json.load(fp)
with open('v100_time.json', 'r') as fp:
    v100_time_only = json.load(fp)

job_list = []
final1_inverse = []
k80 = []
v100 = []
final1_inverse_K80 = []
final1_inverse_V100 = []
final1_inverse_overhead = []
final1_inverse_epoch_waste = []
final1_inverse_ovhd_a = []
final1_inverse_ovhd_b = []
final1_inverse_ovhd_c = []
final1_inverse_ovhd_d = []
final1_inverse_k80_1st = []
final1_inverse_v100_1st = []

final1_inverse_num_mig = []

speedup = []
epoch_num = []
k80_time = []
v100_time = []

for i in range(50):
    job = str(i+1)
    job_list.append('job'+job)
    final1_inverse.append(final1_inverse_only[job])
    k80.append(k80_only[job])
    v100.append(v100_only[job])
    final1_inverse_K80.append(final1_inverse_K80_only[job])
    final1_inverse_V100.append(final1_inverse_V100_only[job])
    final1_inverse_overhead.append(final1_inverse_overhead_only[job])
    final1_inverse_epoch_waste.append(final1_inverse_epoch_waste_only['job'+job])
    final1_inverse_num_mig.append(final1_inverse_num_mig_only[job])
    if len(final1_inverse_ovhd_a_only[job]) > 0:
        final1_inverse_ovhd_a.append(int(np.mean(final1_inverse_ovhd_a_only[job])))
    else:
        final1_inverse_ovhd_a.append(0)
    if len(final1_inverse_ovhd_b_only[job]) > 0:
        final1_inverse_ovhd_b.append(int(np.mean(final1_inverse_ovhd_b_only[job])))
    else:
        final1_inverse_ovhd_b.append(0)
    if len(final1_inverse_ovhd_c_only[job]) > 0:
        final1_inverse_ovhd_c.append(int(np.mean(final1_inverse_ovhd_c_only[job])))
    else:
        final1_inverse_ovhd_c.append(0)
    if len(final1_inverse_ovhd_d_only[job]) > 0:
        final1_inverse_ovhd_d.append(int(np.mean(final1_inverse_ovhd_d_only[job])))
    else:
        final1_inverse_ovhd_d.append(0)
    if len(final1_inverse_k80_1st_only[job]) > 0:
        final1_inverse_k80_1st.append(int(np.mean(final1_inverse_k80_1st_only[job])))
    else:
        final1_inverse_k80_1st.append(0)
    if len(final1_inverse_v100_1st_only[job]) > 0:
        final1_inverse_v100_1st.append(int(np.mean(final1_inverse_v100_1st_only[job])))
    else:
        final1_inverse_v100_1st.append(0)

    speedup.append(round(speedup_only[job],2))
    epoch_num.append(epoch_num_only[job])
    k80_time.append(k80_time_only[job])
    v100_time.append(v100_time_only[job])

job_list = np.asarray(job_list)
final1_inverse = np.asarray(final1_inverse)
k80 = np.asarray(k80)
v100 = np.asarray(v100)
final1_inverse_K80 = np.asarray(final1_inverse_K80)
final1_inverse_V100 = np.asarray(final1_inverse_V100)
final1_inverse_overhead = np.asarray(final1_inverse_overhead)
final1_inverse_epoch_waste = np.asarray(final1_inverse_epoch_waste)

final1_inverse_num_mig = np.asarray(final1_inverse_num_mig)
final1_inverse_ovhd_a = np.asarray(final1_inverse_ovhd_a)
final1_inverse_ovhd_b = np.asarray(final1_inverse_ovhd_b)
final1_inverse_ovhd_c = np.asarray(final1_inverse_ovhd_c)
final1_inverse_ovhd_d = np.asarray(final1_inverse_ovhd_d)
final1_inverse_k80_1st = np.asarray(final1_inverse_k80_1st)
final1_inverse_v100_1st = np.asarray(final1_inverse_v100_1st)

speedup = np.asarray(speedup)
epoch_num = np.asarray(epoch_num)
k80_time = np.asarray(k80_time)
v100_time = np.asarray(v100_time)

rows = zip(job_list, epoch_num, k80, v100, final1_inverse, final1_inverse_K80, final1_inverse_V100, final1_inverse_overhead, final1_inverse_epoch_waste, final1_inverse_num_mig,
final1_inverse_ovhd_a, final1_inverse_ovhd_b, final1_inverse_ovhd_c,final1_inverse_ovhd_d,final1_inverse_k80_1st,k80_time, final1_inverse_v100_1st,v100_time, speedup)
with open('comparison.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

#np.savetxt('comparison.csv', (job_list, k80, v100, final1_inverse, final1_inverse_K80, final1_inverse_V100, final1_inverse_overhead, final1_inverse_num_mig,
#total_time, save_time, load_time), fmt='%s')

