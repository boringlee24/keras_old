import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('oracle/logs/oracle_JCT.json', 'r') as fp:
    oracle_only = json.load(fp)
with open('unaware/logs/unaware_JCT.json', 'r') as fp:
    unaware_only = json.load(fp)

oracle = []
unaware = []

for i in range(50):
    job = str(i+1)
    oracle.append(oracle_only[job])
    unaware.append(unaware_only[job])

speedup = np.mean(unaware) / np.mean(oracle)

norm = []
for i in range(len(unaware)):
    job = str(i+1)
    norm.append(round(unaware_only[job]/oracle_only[job], 1))

avg = np.mean(norm)
print(list(enumerate(norm)))

sorted_unaware = sorted(unaware_only.items(), key=operator.itemgetter(1))
keys = []
JCTs = []
norm_JCTs = []

for item in sorted_unaware:
    if item[0] != 'average':
        keys.append(item[0])
        JCTs.append(item[1])
        norm_JCTs.append(norm[int(item[0]) - 1])


rows = zip(np.asarray(keys), np.asarray(JCTs), np.asarray(norm_JCTs))
with open('abc.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


