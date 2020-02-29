import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'final2'
JCT_log = './final2/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

final2_time = []
for i in range(50):
    job = str(i+1)
    final2_time.append(JCT[job])

final2_time = np.array(final2_time, dtype=np.float)

overhead_log = './final2/logs/' + testcase + '_overhead.json'
epoch_waste_log = './final2/logs/' + testcase + '_epoch_waste.json'

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
final2_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
final2_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

final2_running = final2_time - final2_overhead - final2_epoch_waste
#final2_overhead = final2_overhead / final2_time * 100
avg = np.average(final2_overhead)
print(f'average overhead {avg}s')
#final2_epoch_waste = final2_epoch_waste / final2_time * 100
avg = np.average(final2_epoch_waste)
print(f'average epoch_waste {avg}s')
#final2_running = final2_running / final2_time * 100
avg = np.average(final2_running)
print(f'average running {avg}s')

x = np.arange(len(final2_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, final2_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, final2_epoch_waste, width, label='epoch waste time', bottom=final2_overhead)
rects3 = ax.bar(x, final2_running, width, label='running time', bottom=final2_epoch_waste+final2_overhead)

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

       
