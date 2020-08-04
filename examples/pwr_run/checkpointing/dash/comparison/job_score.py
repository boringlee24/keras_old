import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('v100_only_JCT.json', 'r') as fp:
    V100_time = json.load(fp)
with open('k80_only_JCT.json', 'r') as fp:
    K80_time = json.load(fp)

score = []

for i in range(100):
    job = str(i+1)
    if i < 50:
        k80_total_time = K80_time[job]
        v100_total_time = V100_time[job]
    elif i < 100:
        joob = str(i+1-50)
        k80_total_time = K80_time[joob]
        v100_total_time = V100_time[joob]
    score.append(round((k80_total_time - v100_total_time) / k80_total_time, 4))

with open('score.json', 'w') as fp:
    json.dump(score, fp, indent=4)


