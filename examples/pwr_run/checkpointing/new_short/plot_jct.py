import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'random2'
JCT_log = './random2/logs/' + testcase + '_JCT.json'
PJCT_log = './random2/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

random2_time = []
for i in range(50):
    job = str(i+1)
    if job in PJCT:
        random2_time.append(PJCT[job])
    else:
        random2_time.append(JCT[job])

random2_time = np.array(random2_time, dtype=np.float)

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

testcase = 'max_par'
JCT_log = './max_par/logs/' + testcase + '_JCT.json'
PJCT_log = './max_par/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

max_par_time = []
for i in range(50):
    job = str(i+1)
    if job in PJCT:
        max_par_time.append(PJCT[job])
    else:
        max_par_time.append(JCT[job])

max_par_time = np.array(max_par_time, dtype=np.float)

x = np.arange(len(random2_time)) + 1  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x - width, random2_time/random2_time, width, label='random/min migration')
rects2 = ax.bar(x, max_pwr_time/random2_time, width, label='max power')
rects3 = ax.bar(x + width, max_par_time/random2_time, width, label='max parameter')

ax.set_ylabel('PJCT normalized')
ax.set_xlabel('job')
ax.set_title('PJCT of the 50-job trace normalized to random')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()

plt.savefig('results.png')

# make sure all jobs are recorded
##if len(time) != 50:
##    print('Error. Not all jobs recorded')
##else:
##    avg = sum(time) / len(time)
##    print(str(avg))
        
