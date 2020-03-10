import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open('num_mig.json', 'r') as fp:
    num_mig = json.load(fp)
with open('overhead.json', 'r') as fp:
    overhead = json.load(fp)

overhead_profile = []
for i in range(50):
    job = str(i+1)
    if num_mig[job] > 0:
        overhead_profile.append(overhead[job] / num_mig[job])
    else:
        overhead_profile.append(0)

overhead = overhead_profile
x = np.arange(len(overhead)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, overhead, width, label='overhead time per migration')

ax.set_ylabel('time(s)')
ax.set_xlabel('job')
ax.set_title('overhead time per migration for all jobs. 0 means it\'s not profiled')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#pdb.set_trace()
plt.savefig('overhead.png')

       
