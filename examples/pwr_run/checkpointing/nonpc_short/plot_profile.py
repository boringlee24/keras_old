import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'feedback'
JCT_log = './feedback/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

feedback_time = []
for i in range(50):
    job = str(i+1)
    feedback_time.append(JCT[job])

feedback_time = np.array(feedback_time, dtype=np.float)

overhead_log = './feedback/logs/' + testcase + '_overhead.json'
epoch_waste_log = './feedback/logs/' + testcase + '_epoch_waste.json'

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
feedback_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
feedback_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

feedback_running = feedback_time - feedback_overhead - feedback_epoch_waste
#feedback_overhead = feedback_overhead / feedback_time * 100
avg = np.average(feedback_overhead)
print(f'average overhead {avg}s')
#feedback_epoch_waste = feedback_epoch_waste / feedback_time * 100
avg = np.average(feedback_epoch_waste)
print(f'average epoch_waste {avg}s')
#feedback_running = feedback_running / feedback_time * 100
avg = np.average(feedback_running)
print(f'average running {avg}s')

x = np.arange(len(feedback_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, feedback_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, feedback_epoch_waste, width, label='epoch waste time', bottom=feedback_overhead)
rects3 = ax.bar(x, feedback_running, width, label='running time', bottom=feedback_epoch_waste+feedback_overhead)

ax.set_ylabel('JCT(%)')
ax.set_xlabel('job')
ax.set_title('JCT of the 50-job trace composed of time profile in percentage')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#plt.show()
pdb.set_trace()
plt.savefig('time_profile_pct.png')

       
