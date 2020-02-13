import glob
import json
import pdb

testcase = 'random2'
JCT_log = './random2/logs/' + testcase + '_JCT.json'
PJCT_log = './random2/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

time = []
for job in JCT:
    if job != 'average':
        if job in PJCT:
            time.append(PJCT[job])
        else:
            time.append(JCT[job])

# make sure all jobs are recorded
if len(time) != 50:
    print('Error. Not all jobs recorded')
else:
    avg = sum(time) / len(time)
    print(str(avg))
        
