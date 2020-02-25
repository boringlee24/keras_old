import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'v100_only'
JCT_log = './v100_only/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

v100_only_time = []
for i in range(50):
    job = str(i+1)
    v100_only_time.append(JCT[job])

v100_only_time = np.array(v100_only_time, dtype=np.float)

overhead_log = './v100_only/logs/' + testcase + '_overhead.json'
epoch_waste_log = './v100_only/logs/' + testcase + '_epoch_waste.json'

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
v100_only_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
v100_only_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

v100_only_running = v100_only_time - v100_only_overhead - v100_only_epoch_waste
#v100_only_overhead = v100_only_overhead / v100_only_time * 100
avg = np.average(v100_only_overhead)
print(f'average overhead {avg}s')
#v100_only_epoch_waste = v100_only_epoch_waste / v100_only_time * 100
avg = np.average(v100_only_epoch_waste)
print(f'average epoch_waste {avg}s')
#v100_only_running = v100_only_running / v100_only_time * 100
avg = np.average(v100_only_running)
print(f'average running {avg}s')

x = np.arange(len(v100_only_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, v100_only_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, v100_only_epoch_waste, width, label='epoch waste time', bottom=v100_only_overhead)
rects3 = ax.bar(x, v100_only_running, width, label='running time', bottom=v100_only_epoch_waste+v100_only_overhead)

ax.set_ylabel('JCT(%)')
ax.set_xlabel('job')
ax.set_title('JCT of the 50-job trace composed of time profile in percentage')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#plt.show()
#pdb.set_trace()
plt.savefig('./plots/' + testcase + '_profile_pct.png')

       
