import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

#with open('../oracle/logs/oracle_JCT.json', 'r') as fp:
#    oracle_only = json.load(fp)
with open('../unaware/logs/unaware_JCT.json', 'r') as fp:
    baseline_only = json.load(fp)
with open('../random/logs/random_JCT.json', 'r') as fp:
    baseline_plus_only = json.load(fp)
with open('../final4_3level_new/logs/final4_3level_JCT.json', 'r') as fp:
    scheme_only = json.load(fp)
with open('../feedback_fair/logs/feedback_fair_JCT.json', 'r') as fp:
    feedback_only = json.load(fp)

with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)
with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)

baseline = []
baseline_plus = []
scheme = []
feedback = []

k80 = []
v100 = []

for i in range(len(baseline_only)-1):
    job = str(i+1)
    baseline.append(baseline_only[job])
    baseline_plus.append(baseline_plus_only[job])
    scheme.append(scheme_only[job])
    feedback.append(feedback_only[job])

    if i < 50:
        k80.append(k80_only[job])
        v100.append(v100_only[job])
    elif i < 100:
        joob = str(i+1-50)
        k80.append(k80_only[joob])
        v100.append(v100_only[joob])


baseline = np.asarray(baseline)
baseline_plus = np.asarray(baseline_plus)
scheme = np.asarray(scheme)
feedback = np.asarray(feedback)

cols = zip(baseline)
with open('JCT/baseline_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(baseline_plus)
with open('JCT/baseline_plus_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme)
with open('JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(feedback)
with open('JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

cols = zip(k80)
with open('JCT/k80_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100)
with open('JCT/v100_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

#########################################

norm_baseline = []
norm_baseline_plus = []
norm_scheme = []
norm_feedback = []

v100_baseline = []
v100_baseline_plus = []
v100_scheme = []
v100_feedback = []

k80_baseline = []
k80_baseline_plus = []
k80_scheme = []
k80_feedback = []

for i in range(len(baseline)):
    job = str(i+1)
    norm_baseline.append(round(baseline_only[job]/baseline_only[job], 2))
    norm_baseline_plus.append(round(baseline_only[job]/baseline_plus_only[job], 2))
    norm_scheme.append(round(baseline_only[job]/scheme_only[job], 2))
    norm_feedback.append(round(baseline_only[job]/feedback_only[job], 2))

    if i < 50:
        v100_baseline.append(round(v100_only[job]/baseline_only[job], 2))   
        v100_baseline_plus.append(round(v100_only[job]/baseline_plus_only[job], 2))
        v100_scheme.append(round(v100_only[job]/scheme_only[job], 2))
        v100_feedback.append(round(v100_only[job]/feedback_only[job], 2))

        k80_baseline.append(round(k80_only[job]/baseline_only[job], 2))   
        k80_baseline_plus.append(round(k80_only[job]/baseline_plus_only[job], 2))
        k80_scheme.append(round(k80_only[job]/scheme_only[job], 2))
        k80_feedback.append(round(k80_only[job]/feedback_only[job], 2))

    elif i < 100:
        joob = str(i+1-50)
        v100_joob = v100_only[joob]
        k80_joob = k80_only[joob]
        v100_baseline.append(round(v100_joob/baseline_only[job], 2))   
        v100_baseline_plus.append(round(v100_joob/baseline_plus_only[job], 2))
        v100_scheme.append(round(v100_joob/scheme_only[job], 2))
        v100_feedback.append(round(v100_joob/feedback_only[job], 2))

        k80_baseline.append(round(k80_joob/baseline_only[job], 2))   
        k80_baseline_plus.append(round(k80_joob/baseline_plus_only[job], 2))
        k80_scheme.append(round(k80_joob/scheme_only[job], 2))
        k80_feedback.append(round(k80_joob/feedback_only[job], 2))

#pdb.set_trace()
cols = zip(norm_baseline)
with open('norm_JCT/baseline_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_baseline_plus)
with open('norm_JCT/baseline_plus_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_scheme)
with open('norm_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_feedback)
with open('norm_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

cols = zip(v100_baseline)
with open('v100_JCT/baseline_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_baseline_plus)
with open('v100_JCT/baseline_plus_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_scheme)
with open('v100_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_feedback)
with open('v100_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

cols = zip(k80_baseline)
with open('k80_JCT/baseline_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_baseline_plus)
with open('k80_JCT/baseline_plus_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_scheme)
with open('k80_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_feedback)
with open('k80_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)


