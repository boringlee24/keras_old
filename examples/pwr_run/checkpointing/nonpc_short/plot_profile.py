import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'final1'
JCT_log = './final1/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

final1_time = []
for i in range(50):
    job = str(i+1)
    final1_time.append(JCT[job])

final1_time = np.array(final1_time, dtype=np.float)

overhead_log = './final1/logs/' + testcase + '_overhead.json'
epoch_waste_log = './final1/logs/' + testcase + '_epoch_waste.json'

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
final1_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
final1_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

final1_running = final1_time - final1_overhead - final1_epoch_waste
#final1_overhead = final1_overhead / final1_time * 100
avg = np.average(final1_overhead)
print(f'average overhead {avg}s')
#final1_epoch_waste = final1_epoch_waste / final1_time * 100
avg = np.average(final1_epoch_waste)
print(f'average epoch_waste {avg}s')
#final1_running = final1_running / final1_time * 100
avg = np.average(final1_running)
print(f'average running {avg}s')

x = np.arange(len(final1_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, final1_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, final1_epoch_waste, width, label='epoch waste time', bottom=final1_overhead)
rects3 = ax.bar(x, final1_running, width, label='running time', bottom=final1_epoch_waste+final1_overhead)

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

       
