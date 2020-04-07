import random
import pdb
import json

job_num = 50
job_queue = random.sample(range(1, job_num+1), job_num)

print(job_queue)

with open('job_queue.json', 'w') as fp1:
    json.dump(job_queue, fp1, indent=4)

