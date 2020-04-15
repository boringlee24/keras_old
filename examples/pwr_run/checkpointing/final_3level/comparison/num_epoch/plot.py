import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json

k80_log = '/scratch/li.baol/tsrbrd_log/job_runs/debug/k80_only/'
v100_log = '/scratch/li.baol/tsrbrd_log/job_runs/debug/v100_only/'

num_epoch = []
k80_time = []
k80_dict = {}
v100_time = []
v100_dict = {}
epoch_dict = {}
for i in range(50):
    job = 'job' + str(i+1)
    jobb = job.replace('job','')
    k80_dir = k80_log + job + '/*'
    v100_dir = v100_log + job + '/*'

    k80_dirs = glob.glob(k80_dir)
    tc = k80_dirs[0]
    iterator = EventAccumulator(tc).Reload()
    tag = 'loss'
    wall_time = [t.wall_time for t in iterator.Scalars(tag)]
    num_epoch.append(len(wall_time))
    epoch_dict[jobb] = len(wall_time)

    k80_time.append((wall_time[-1] - wall_time[0]) / (len(wall_time) - 1))
    k80_dict[jobb] = round((wall_time[-1] - wall_time[0]) / (len(wall_time) - 1), 1)

    v100_dirs = glob.glob(v100_dir)
    tc = v100_dirs[0]
    iterator = EventAccumulator(tc).Reload()
    tag = 'loss'
    wall_time = [t.wall_time for t in iterator.Scalars(tag)]
    v100_time.append((wall_time[-1] - wall_time[0]) / (len(wall_time) - 1))
    v100_dict[jobb] = round((wall_time[-1] - wall_time[0]) / (len(wall_time) - 1), 1)

with open('k80_time.json', 'w') as f:
    json.dump(k80_dict, f, indent=4)
with open('v100_time.json', 'w') as f:
    json.dump(v100_dict, f, indent=4)
with open('epoch_num.json', 'w') as f:
    json.dump(epoch_dict, f, indent=4)


pdb.set_trace()

x = np.arange(len(num_epoch)) + 1  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, num_epoch, width, label='number of epochs')
ax.set_ylabel('epoch number')
ax.set_xlabel('job')
ax.set_title('number of epochs for all jobs')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')
fig.tight_layout()
#pdb.set_trace()
plt.savefig('num_epoch.png')

fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, k80_time, width, label='k80 time per epoch')
ax.set_ylabel('time (s)')
ax.set_xlabel('job')
ax.set_title('epoch time for all jobs')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')
fig.tight_layout()
#pdb.set_trace()
plt.savefig('k80_time.png')
 
fig, ax = plt.subplots(figsize=(12,5))
rects1 = ax.bar(x, v100_time, width, label='v100 time per epoch')
ax.set_ylabel('time (s)')
ax.set_xlabel('job')
ax.set_title('epoch time for all jobs')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend()
ax.grid(which='major', axis='y', linestyle='dotted')
fig.tight_layout()
#pdb.set_trace()
plt.savefig('v100_time.png')
     
