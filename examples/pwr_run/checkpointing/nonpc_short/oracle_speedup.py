import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'v100_only'
v100_JCT_log = './v100_only/logs/' + testcase + '_JCT.json'
testcase = 'k80_only'
k80_JCT_log = './k80_only/logs/' + testcase + '_JCT.json'

v100_JCT = {}
with open(v100_JCT_log, 'r') as fp:
    v100_JCT = json.load(fp)
k80_JCT = {}
with open(k80_JCT_log, 'r') as fp:
    k80_JCT = json.load(fp)

speedup_dict = {}
for i in range(50):
    job = str(i+1)
    v100_time = v100_JCT[job]
    k80_time = k80_JCT[job]
    speedup = (k80_time - v100_time) / k80_time
    speedup_dict[job] = speedup

with open('speedup.json', 'w') as fp:
    json.dump(speedup_dict, fp, sort_keys=True, indent=4)
    

