import glob
import json
import pdb

testcase = 'max_par'
JCT_log = './max_par/logs/' + testcase + '_JCT.json'
PJCT_log = './max_par/logs/' + testcase + '_PJCT.json'

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
        
