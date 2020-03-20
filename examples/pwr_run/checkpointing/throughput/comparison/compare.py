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
with open('../final2/logs_600_1/final2_JCT.json', 'r') as fp:
    final2_only = json.load(fp)
with open('../final1_test/logs/final1_test_JCT.json', 'r') as fp:
    final1_test_only = json.load(fp)
with open('../final1_inverse/logs/final1_inverse_JCT.json', 'r') as fp:
    final1_inverse_only = json.load(fp)
with open('../final2_inverse/logs/final2_inverse_JCT.json', 'r') as fp:
    final2_inverse_only = json.load(fp)
with open('../final3/logs/final3_JCT.json', 'r') as fp:
    final3_only = json.load(fp)
with open('../final4/logs/final4_JCT.json', 'r') as fp:
    final4_only = json.load(fp)

oracle = []
unaware = []
random = []
feedback = []
final1 = []
final2 = []
final1_test = []
final1_inverse = []
final2_inverse = []
final3 = []
final4 = []

for i in range(50):
    job = str(i+1)
    oracle.append(oracle_only[job])
    unaware.append(unaware_only[job])
    random.append(random_only[job])
    feedback.append(feedback_only[job])
    final1.append(final1_only[job])
    final2.append(final2_only[job])
    final1_test.append(final1_test_only[job])
    final1_inverse.append(final1_inverse_only[job])
    final2_inverse.append(final2_inverse_only[job])
    final3.append(final3_only[job])
    final4.append(final4_only[job])

speedup = np.mean(unaware) / np.mean(oracle)

norm_oracle = []
norm_random = []
norm_feedback = []
norm_final1 = []
norm_final2 = []
norm_final1_test = []
norm_final1_inverse = []
norm_final2_inverse = []
norm_final3 = []
norm_final4 = []

for i in range(len(unaware)):
    job = str(i+1)
    norm_oracle.append(round(unaware_only[job]/oracle_only[job], 1))
    norm_random.append(round(unaware_only[job]/random_only[job], 1))
    norm_feedback.append(round(unaware_only[job]/feedback_only[job], 1))
    norm_final1.append(round(unaware_only[job]/final1_only[job], 1))
    norm_final2.append(round(unaware_only[job]/final2_only[job], 1))
    norm_final1_test.append(round(unaware_only[job]/final1_test_only[job], 1))
    norm_final1_inverse.append(round(unaware_only[job]/final1_inverse_only[job], 1))
    norm_final2_inverse.append(round(unaware_only[job]/final2_inverse_only[job], 1))
    norm_final3.append(round(unaware_only[job]/final3_only[job], 1))
    norm_final4.append(round(unaware_only[job]/final4_only[job], 1))


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
