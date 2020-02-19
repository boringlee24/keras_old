import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'min_par'
JCT_log = './min_par/logs/' + testcase + '_JCT.json'
PJCT_log = './min_par/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

min_par_time = []
for i in range(50):
    job = str(i+1)
    if job in PJCT:
        min_par_time.append(PJCT[job])
    else:
        min_par_time.append(JCT[job])

min_par_time = np.array(min_par_time, dtype=np.float)

overhead_log = './min_par/logs/' + testcase + '_overhead.json'
epoch_waste_log = './min_par/logs/' + testcase + '_epoch_waste.json'

overhead = {}
epoch_waste = {}
with open(overhead_log, 'r') as fp:
    overhead = json.load(fp)
with open(epoch_waste_log, 'r') as fp:
    epoch_waste = json.load(fp)

overhead_time = []
for i in range(50):
    job = str(i+1)
    overhead_time.append(overhead[job])
min_par_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
min_par_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

min_par_running = min_par_time - min_par_overhead - min_par_epoch_waste
min_par_overhead = min_par_overhead / min_par_time * 100
avg = np.mean(min_par_overhead)
print(f'average overhead {avg}%')
min_par_epoch_waste = min_par_epoch_waste / min_par_time * 100
avg = np.mean(min_par_epoch_waste)
print(f'average epoch_waste {avg}%')
min_par_running = min_par_running / min_par_time * 100
avg = np.mean(min_par_running)
print(f'average running {avg}%')

x = np.arange(len(min_par_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, min_par_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, min_par_epoch_waste, width, label='epoch waste time', bottom=min_par_overhead)
rects3 = ax.bar(x, min_par_running, width, label='running time', bottom=min_par_epoch_waste+min_par_overhead)

ax.set_ylabel('PJCT(%)')
ax.set_xlabel('job')
ax.set_title('PJCT of the 50-job trace composed of time profile in percentage')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
plt.show()
pdb.set_trace()
plt.savefig('time_profile_pct.png')

       
