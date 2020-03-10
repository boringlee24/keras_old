import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)
with open('oracle_JCT.json', 'r') as fp:
    oracle_only = json.load(fp)
with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)

oracle = []
k80 = []
v100 = []

for i in range(50):
    job = str(i+1)
    oracle.append(oracle_only[job])
    k80.append(k80_only[job])
    v100.append(v100_only[job])

norm = []
for i in range(len(k80)):
    job = str(i+1)
    norm.append(round((k80_only[job] - oracle_only[job]) / k80_only[job] * 100, 1))

avg = np.mean(norm)
print(enumerate(norm))
pdb.set_trace()

x = np.arange(len(k80)) + 1  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x-width, k80, width, label='k80')
rects1 = ax.bar(x, oracle, width, label='oracle')
rects1 = ax.bar(x+width, v100, width, label='v100')

ax.set_ylabel('time(s)')
ax.set_xlabel('job')
ax.set_title('Comparison of oracle scheme with worst performance and best performance per job')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')

fig.tight_layout()
#plt.show()
#pdb.set_trace()
plt.savefig('oracle.png')

       
