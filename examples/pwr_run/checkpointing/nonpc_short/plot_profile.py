import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'oracle'
JCT_log = './oracle/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

oracle_time = []
for i in range(50):
    job = str(i+1)
    oracle_time.append(JCT[job])

oracle_time = np.array(oracle_time, dtype=np.float)

overhead_log = './oracle/logs/' + testcase + '_overhead.json'
epoch_waste_log = './oracle/logs/' + testcase + '_epoch_waste.json'

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
oracle_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
oracle_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

oracle_running = oracle_time - oracle_overhead - oracle_epoch_waste
#oracle_overhead = oracle_overhead / oracle_time * 100
avg = np.average(oracle_overhead)
print(f'average overhead {avg}s')
#oracle_epoch_waste = oracle_epoch_waste / oracle_time * 100
avg = np.average(oracle_epoch_waste)
print(f'average epoch_waste {avg}s')
#oracle_running = oracle_running / oracle_time * 100
avg = np.average(oracle_running)
print(f'average running {avg}s')

x = np.arange(len(oracle_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, oracle_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, oracle_epoch_waste, width, label='epoch waste time', bottom=oracle_overhead)
rects3 = ax.bar(x, oracle_running, width, label='running time', bottom=oracle_epoch_waste+oracle_overhead)

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

       
