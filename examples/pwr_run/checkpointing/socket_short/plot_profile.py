import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'max_pwr'
JCT_log = './max_pwr/logs/' + testcase + '_JCT.json'
PJCT_log = './max_pwr/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

max_pwr_time = []
for i in range(50):
    job = str(i+1)
    if job in PJCT:
        max_pwr_time.append(PJCT[job])
    else:
        max_pwr_time.append(JCT[job])

max_pwr_time = np.array(max_pwr_time, dtype=np.float)

overhead_log = './max_pwr/logs/' + testcase + '_overhead.json'
epoch_waste_log = './max_pwr/logs/' + testcase + '_epoch_waste.json'

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
max_pwr_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
max_pwr_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

max_pwr_running = max_pwr_time - max_pwr_overhead - max_pwr_epoch_waste
#max_pwr_overhead = max_pwr_overhead / max_pwr_time * 100
avg = np.average(max_pwr_overhead)
print(f'average overhead {avg}s')
#max_pwr_epoch_waste = max_pwr_epoch_waste / max_pwr_time * 100
avg = np.average(max_pwr_epoch_waste)
print(f'average epoch_waste {avg}s')
#max_pwr_running = max_pwr_running / max_pwr_time * 100
avg = np.average(max_pwr_running)
print(f'average running {avg}s')

x = np.arange(len(max_pwr_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, max_pwr_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, max_pwr_epoch_waste, width, label='epoch waste time', bottom=max_pwr_overhead)
rects3 = ax.bar(x, max_pwr_running, width, label='running time', bottom=max_pwr_epoch_waste+max_pwr_overhead)

ax.set_ylabel('PJCT(%)')
ax.set_xlabel('job')
ax.set_title('PJCT of the 50-job trace composed of time profile in percentage')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#plt.show()
pdb.set_trace()
plt.savefig('time_profile_pct.png')

       
