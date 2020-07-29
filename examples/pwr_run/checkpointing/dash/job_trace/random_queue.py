import random
import pdb
import json

#job_num = 30
#job_queue = random.sample(range(4, job_num+1), job_num-3)
#job_queue_2gpu = random.sample(range(1, 4), 3)
#job_queue = job_queue_2gpu + job_queue
#
#print(job_queue)
#
#with open('job_queue_30.json', 'w') as fp1:
#    json.dump(job_queue, fp1, indent=4)

unaware_log = '/home/li.baol/GIT/keras_old/examples/pwr_run/checkpointing/dash/unaware/logs/unaware_K80_time.json'

with open(unaware_log) as f:
    unaware = json.load(f)

job_queue_K = []
job_queue_V = []
for i in range(100):
    job = str(i+1)
    if unaware[job] == 0:
        job_queue_K.append(job)
    else:
        job_queue_V.append(job)

with open('job_queue_K.json', 'w') as fp1:
    json.dump(job_queue_K, fp1, indent=4)
with open('job_queue_V.json', 'w') as fp1:
    json.dump(job_queue_V, fp1, indent=4)

