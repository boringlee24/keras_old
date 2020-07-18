import pdb
import json

num_epochs = {}

# read from job8.py to job100.py
for i in range(8, 101):
    i = str(i)
    job = 'job'+i+'.py'

    f = open(job, "r")
    lines = f.readlines()
    for line in lines:
        if 'total_epochs = ' in line:
            epoch = line.split(' = ')[1].split('\n')[0]
    num_epochs['job'+i] = epoch

with open('num_epochs.json', 'w') as f:
    json.dump(num_epochs, f, indent=4)

