import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open('speedup.json', 'r') as fp:
    speedup = json.load(fp)

speedup_profile = []
for i in range(50):
    job = str(i+1)
    speedup_profile.append(speedup[job])

speedup = speedup_profile
x = np.arange(len(speedup)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, speedup, width, label='speedup')

ax.set_ylabel('speedup')
ax.set_xlabel('job')
ax.set_title('Speedup scaled between 0 - 1. Calculated as (k80_time - v100_time) / k80_time')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#pdb.set_trace()
plt.savefig('speedup.png')

       
