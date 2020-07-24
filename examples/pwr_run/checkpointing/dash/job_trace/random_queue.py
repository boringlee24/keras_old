import random
import pdb
import json

job_num = 100
job_queue = random.sample(range(8, job_num+1), job_num-7)
job_queue_2gpu = random.sample(range(1, 8), 7)
job_queue = job_queue_2gpu + job_queue

print(job_queue)

with open('job_queue_100.json', 'w') as fp1:
    json.dump(job_queue, fp1, indent=4)

