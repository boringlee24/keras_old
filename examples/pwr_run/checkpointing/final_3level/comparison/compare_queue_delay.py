import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('../unaware/logs/unaware_queue_delay.json', 'r') as fp:
    baseline_only = json.load(fp)
with open('../random/logs/random_queue_delay.json', 'r') as fp:
    baseline_plus_only = json.load(fp)
with open('../final4_3level_new/logs/final4_3level_queue_delay.json', 'r') as fp:
    scheme_only = json.load(fp)
with open('../feedback_fair/logs/feedback_fair_JCT.json', 'r') as fp:
    feedback_only = json.load(fp)

baseline = []
baseline_plus = []
scheme = []
feedback = []

for i in range(len(baseline_only)-1):
    job = str(i+1)
    baseline.append(baseline_only[job])
    baseline_plus.append(baseline_plus_only[job])
    scheme.append(scheme_only[job])
    feedback.append(feedback_only[job])

baseline = np.asarray(baseline)
baseline_plus = np.asarray(baseline_plus)
scheme = np.asarray(scheme)
feedback = np.asarray(feedback)

cols = zip(baseline)
with open('queue_delay/baseline_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(baseline_plus)
with open('queue_delay/baseline_plus_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme)
with open('queue_delay/scheme_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(feedback)
with open('queue_delay/feedback_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

