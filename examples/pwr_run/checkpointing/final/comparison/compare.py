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
    unaware_only = json.load(fp)
with open('../random/logs/random_JCT.json', 'r') as fp:
    random_only = json.load(fp)
with open('../feedback_inverse/logs/feedback_inverse_JCT.json', 'r') as fp:
    feedback_inverse_only = json.load(fp)
with open('../final4_new/logs_0.05thres/final4_new_JCT.json', 'r') as fp:
    scheme_new_only = json.load(fp)
with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)
with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)

oracle = []
unaware = []
random = []
feedback_inverse = []
final1 = []
final2 = []
final1_test = []
final1_inverse = []
final2_inverse = []
final3 = []
final4 = []
final5 = []
scheme_new = []
k80 = []
v100 = []

for i in range(len(unaware_only)-1):
    job = str(i+1)
#    oracle.append(oracle_only[job])
    unaware.append(unaware_only[job])
    random.append(random_only[job])
    feedback_inverse.append(feedback_inverse_only[job])
#    final1.append(final1_only[job])
#    final2.append(final2_only[job])
#    final1_test.append(final1_test_only[job])
#    final1_inverse.append(final1_inverse_only[job])
#    final2_inverse.append(final2_inverse_only[job])
#    final3.append(final3_only[job])
#    final4.append(final4_only[job])
#    final5.append(final5_only[job])
    scheme_new.append(scheme_new_only[job])
    if i < 50:
        k80.append(k80_only[job])
        v100.append(v100_only[job])
    elif i < 100:
        joob = str(i+1-50)
        k80.append(k80_only[joob])
        v100.append(v100_only[joob])


unaware = np.asarray(unaware)
random = np.asarray(random)
feedback_inverse = np.asarray(feedback_inverse)
scheme_new = np.asarray(scheme_new)

cols = zip(unaware)
with open('JCT/unaware_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(random)
with open('JCT/random_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(feedback_inverse)
with open('JCT/feedback_inverse_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_new)
with open('JCT/scheme_new_jct.csv', 'w') as f:
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

norm_oracle = []
norm_random = []
norm_feedback_inverse = []
norm_final1 = []
norm_final2 = []
norm_final1_test = []
norm_final1_inverse = []
norm_final2_inverse = []
norm_final3 = []
norm_final4 = []
norm_final5 = []
norm_scheme_new = []

v100_unaware = []
v100_random = []
v100_feedback_inverse = []
v100_scheme_new = []
k80_unaware = []
k80_random = []
k80_feedback_inverse = []
k80_scheme_new = []

for i in range(len(unaware)):
    job = str(i+1)
#    norm_oracle.append(round(unaware_only[job]/oracle_only[job], 2))
    norm_random.append(round(unaware_only[job]/random_only[job], 2))
    norm_feedback_inverse.append(round(unaware_only[job]/feedback_inverse_only[job], 2))
#    norm_final1.append(round(unaware_only[job]/final1_only[job], 2))
#    norm_final2.append(round(unaware_only[job]/final2_only[job], 2))
#    norm_final1_test.append(round(unaware_only[job]/final1_test_only[job], 2))
#    norm_final1_inverse.append(round(unaware_only[job]/final1_inverse_only[job], 2))
#    norm_final2_inverse.append(round(unaware_only[job]/final2_inverse_only[job], 2))
#    norm_final3.append(round(unaware_only[job]/final3_only[job], 2))
#    norm_final4.append(round(unaware_only[job]/final4_only[job], 2))
#    norm_final5.append(round(unaware_only[job]/final5_only[job], 2))
    norm_scheme_new.append(round(unaware_only[job]/scheme_new_only[job], 2))

    if i < 50:
        v100_unaware.append(round(v100_only[job]/unaware_only[job], 2))   
        v100_random.append(round(v100_only[job]/random_only[job], 2))
        v100_feedback_inverse.append(round(v100_only[job]/feedback_inverse_only[job], 2))
        v100_scheme_new.append(round(v100_only[job]/scheme_new_only[job], 2))
        k80_unaware.append(round(k80_only[job]/unaware_only[job], 2))   
        k80_random.append(round(k80_only[job]/random_only[job], 2))
        k80_feedback_inverse.append(round(k80_only[job]/feedback_inverse_only[job], 2))
        k80_scheme_new.append(round(k80_only[job]/scheme_new_only[job], 2))
    elif i < 100:
        joob = str(i+1-50)
        v100_joob = v100_only[joob]
        k80_joob = k80_only[joob]
        v100_unaware.append(round(v100_joob/unaware_only[job], 2))   
        v100_random.append(round(v100_joob/random_only[job], 2))
        v100_feedback_inverse.append(round(v100_joob/feedback_inverse_only[job], 2))
        v100_scheme_new.append(round(v100_joob/scheme_new_only[job], 2))
        k80_unaware.append(round(k80_joob/unaware_only[job], 2))   
        k80_random.append(round(k80_joob/random_only[job], 2))
        k80_feedback_inverse.append(round(k80_joob/feedback_inverse_only[job], 2))
        k80_scheme_new.append(round(k80_joob/scheme_new_only[job], 2))

cols = zip(norm_random)
with open('norm_JCT/random_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_feedback_inverse)
with open('norm_JCT/feedback_inverse_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_scheme_new)
with open('norm_JCT/scheme_new_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

cols = zip(v100_unaware)
with open('v100_JCT/unaware_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_random)
with open('v100_JCT/random_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_feedback_inverse)
with open('v100_JCT/feedback_inverse_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_scheme_new)
with open('v100_JCT/scheme_new_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_unaware)
with open('k80_JCT/unaware_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_random)
with open('k80_JCT/random_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_feedback_inverse)
with open('k80_JCT/feedback_inverse_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_scheme_new)
with open('k80_JCT/scheme_new_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

####################################################

with open('../unaware/logs/unaware_queue_delay.json', 'r') as fp:
    unaware_only = json.load(fp)
with open('../random/logs/random_queue_delay.json', 'r') as fp:
    random_only = json.load(fp)
with open('../feedback_inverse/logs/feedback_inverse_queue_delay.json', 'r') as fp:
    feedback_inverse_only = json.load(fp)
with open('../final4_new/logs_0.05thres/final4_new_queue_delay.json', 'r') as fp:
    scheme_new_only = json.load(fp)

unaware = []
random = []
feedback_inverse = []
scheme_new = []
for i in range(len(unaware_only)-1):
    job = str(i+1)
    unaware.append(unaware_only[job])
    random.append(random_only[job])
    feedback_inverse.append(feedback_inverse_only[job])
    scheme_new.append(scheme_new_only[job])

unaware = np.asarray(unaware)
random = np.asarray(random)
feedback_inverse = np.asarray(feedback_inverse)
scheme_new = np.asarray(scheme_new)

cols = zip(unaware)
with open('queue_delay/unaware_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(random)
with open('queue_delay/random_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(feedback_inverse)
with open('queue_delay/feedback_inverse_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_new)
with open('queue_delay/scheme_new_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

