import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('../oracle/logs/oracle_JCT.json', 'r') as fp:
    oracle_only = json.load(fp)
with open('../unaware/logs/unaware_JCT.json', 'r') as fp:
    unaware_only = json.load(fp)
with open('../random/logs/random_JCT.json', 'r') as fp:
    random_only = json.load(fp)
with open('../feedback/logs/feedback_JCT.json', 'r') as fp:
    feedback_only = json.load(fp)
with open('../final1/logs/final1_JCT.json', 'r') as fp:
    final1_only = json.load(fp)
with open('../final2/logs/final2_JCT.json', 'r') as fp:
    final2_only = json.load(fp)

oracle = []
unaware = []
random = []
feedback = []
final1 = []
final2 = []

for i in range(50):
    job = str(i+1)
    oracle.append(oracle_only[job])
    unaware.append(unaware_only[job])
    random.append(random_only[job])
    feedback.append(feedback_only[job])
    final1.append(final1_only[job])
    final2.append(final2_only[job])

speedup = np.mean(unaware) / np.mean(oracle)

norm_oracle = []
norm_random = []
norm_feedback = []
norm_final1 = []
norm_final2 = []

for i in range(len(unaware)):
    job = str(i+1)
    norm_oracle.append(round(unaware_only[job]/oracle_only[job], 1))
    norm_random.append(round(unaware_only[job]/random_only[job], 1))
    norm_feedback.append(round(unaware_only[job]/feedback_only[job], 1))
    norm_final1.append(round(unaware_only[job]/final1_only[job], 1))
    norm_final2.append(round(unaware_only[job]/final2_only[job], 1))


avg = np.mean(norm_oracle)

pdb.set_trace()
print()
#
#print(list(enumerate(norm)))
#
#sorted_unaware = sorted(unaware_only.items(), key=operator.itemgetter(1))
#keys = []
#JCTs = []
#norm_JCTs = []
#
#for item in sorted_unaware:
#    if item[0] != 'average':
#        keys.append(item[0])
#        JCTs.append(item[1])
#        norm_JCTs.append(norm[int(item[0]) - 1])
#
#
#rows = zip(np.asarray(keys), np.asarray(JCTs), np.asarray(norm_JCTs))
#with open('abc.csv', 'w') as f:
#    writer = csv.writer(f)
#    for row in rows:
#        writer.writerow(row)
#
#
