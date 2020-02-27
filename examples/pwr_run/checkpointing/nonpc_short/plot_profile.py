import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'timed_feedback'
JCT_log = './timed_feedback/logs/' + testcase + '_JCT.json'

JCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)

timed_feedback_time = []
for i in range(50):
    job = str(i+1)
    timed_feedback_time.append(JCT[job])

timed_feedback_time = np.array(timed_feedback_time, dtype=np.float)

overhead_log = './timed_feedback/logs/' + testcase + '_overhead.json'
epoch_waste_log = './timed_feedback/logs/' + testcase + '_epoch_waste.json'

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
timed_feedback_overhead = np.array(overhead_time, dtype=np.float)

epoch_waste_time = []
for i in range(50):
    job = 'job' + str(i+1)
    epoch_waste_time.append(epoch_waste[job])
timed_feedback_epoch_waste = np.array(epoch_waste_time, dtype=np.float)

timed_feedback_running = timed_feedback_time - timed_feedback_overhead - timed_feedback_epoch_waste
#timed_feedback_overhead = timed_feedback_overhead / timed_feedback_time * 100
avg = np.average(timed_feedback_overhead)
print(f'average overhead {avg}s')
#timed_feedback_epoch_waste = timed_feedback_epoch_waste / timed_feedback_time * 100
avg = np.average(timed_feedback_epoch_waste)
print(f'average epoch_waste {avg}s')
#timed_feedback_running = timed_feedback_running / timed_feedback_time * 100
avg = np.average(timed_feedback_running)
print(f'average running {avg}s')

x = np.arange(len(timed_feedback_time)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, timed_feedback_overhead, width, label='mechanical overhead time')
rects2 = ax.bar(x, timed_feedback_epoch_waste, width, label='epoch waste time', bottom=timed_feedback_overhead)
rects3 = ax.bar(x, timed_feedback_running, width, label='running time', bottom=timed_feedback_epoch_waste+timed_feedback_overhead)

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

       
