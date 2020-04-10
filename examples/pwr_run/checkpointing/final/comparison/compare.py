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
with open('../feedback_inverse/logs/feedback_inverse_JCT.json', 'r') as fp:
    feedback_only = json.load(fp)
with open('../final4_new/logs/final4_new_JCT.json', 'r') as fp:
    scheme_only = json.load(fp)
##with open('../final4_new2/logs/final4_new2_JCT.json', 'r') as fp:
##    epoch_boundary_only = json.load(fp)
with open('../final5/logs/final5_JCT.json', 'r') as fp:
    start_on_both_only = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_JCT.json', 'r') as fp:
    no_safeguard_only = json.load(fp)
with open('../no_threshold/logs/no_threshold_JCT.json', 'r') as fp:
    no_threshold_only = json.load(fp)
with open('../predict_error/logs/predict_error_JCT.json', 'r') as fp:
    predict_error_only = json.load(fp)

with open('v100_only_JCT.json', 'r') as fp:
    v100_only = json.load(fp)
with open('k80_only_JCT.json', 'r') as fp:
    k80_only = json.load(fp)

oracle = []
baseline = []
baseline_plus = []
feedback = []
final1 = []
final2 = []
final1_test = []
final1_inverse = []
final2_inverse = []
final3 = []
final4 = []
final5 = []
scheme = []
#epoch_boundary = []
start_on_both = []
no_safeguard = []
no_threshold = []
predict_error = []

k80 = []
v100 = []

for i in range(len(baseline_only)-1):
    job = str(i+1)
    baseline.append(baseline_only[job])
    baseline_plus.append(baseline_plus_only[job])
    feedback.append(feedback_only[job])
    scheme.append(scheme_only[job])
    #epoch_boundary.append(epoch_boundary_only[job])
    start_on_both.append(start_on_both_only[job])
    no_safeguard.append(no_safeguard_only[job])
    no_threshold.append(no_threshold_only[job])
    predict_error.append(predict_error_only[job])

    if i < 50:
        k80.append(k80_only[job])
        v100.append(v100_only[job])
    elif i < 100:
        joob = str(i+1-50)
        k80.append(k80_only[joob])
        v100.append(v100_only[joob])


baseline = np.asarray(baseline)
baseline_plus = np.asarray(baseline_plus)
feedback = np.asarray(feedback)
scheme = np.asarray(scheme)
#epoch_boundary = np.asarray(epoch_boundary)
start_on_both = np.asarray(start_on_both)
no_safeguard = np.asarray(no_safeguard)
no_threshold = np.asarray(no_threshold)
predict_error = np.asarray(predict_error)

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
cols = zip(feedback)
with open('JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme)
with open('JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
##cols = zip(epoch_boundary)
##with open('JCT/epoch_boundary_jct.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
cols = zip(start_on_both)
with open('JCT/start_on_both_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(no_safeguard)
with open('JCT/no_safeguard_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(no_threshold)
with open('JCT/no_threshold_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(predict_error)
with open('JCT/predict_error_jct.csv', 'w') as f:
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
norm_baseline = []
norm_baseline_plus = []
norm_feedback = []
norm_final1 = []
norm_final2 = []
norm_final1_test = []
norm_final1_inverse = []
norm_final2_inverse = []
norm_final3 = []
norm_final4 = []
norm_final5 = []
norm_scheme = []
#norm_epoch_boundary = []
norm_start_on_both = []
norm_no_safeguard = []
norm_no_threshold = []
norm_predict_error = []

v100_baseline = []
v100_baseline_plus = []
v100_feedback = []
v100_scheme = []
#v100_epoch_boundary = []
v100_start_on_both = []
v100_no_safeguard = []
v100_no_threshold = []
v100_predict_error = []

k80_baseline = []
k80_baseline_plus = []
k80_feedback = []
k80_scheme = []
#k80_epoch_boundary = []
k80_start_on_both = []
k80_no_safeguard = []
k80_no_threshold = []
k80_predict_error = []

for i in range(len(baseline)):
    job = str(i+1)
    norm_baseline.append(round(baseline_only[job]/baseline_only[job], 2))
    norm_baseline_plus.append(round(baseline_only[job]/baseline_plus_only[job], 2))
    norm_feedback.append(round(baseline_only[job]/feedback_only[job], 2))
    norm_scheme.append(round(baseline_only[job]/scheme_only[job], 2))
#    norm_epoch_boundary.append(round(baseline_only[job]/epoch_boundary_only[job], 2))
    norm_start_on_both.append(round(baseline_only[job]/start_on_both_only[job], 2))
    norm_no_safeguard.append(round(baseline_only[job]/no_safeguard_only[job], 2))
    norm_no_threshold.append(round(baseline_only[job]/no_threshold_only[job], 2))
    norm_predict_error.append(round(baseline_only[job]/predict_error_only[job], 2))

    if i < 50:
        v100_baseline.append(round(v100_only[job]/baseline_only[job], 2))   
        v100_baseline_plus.append(round(v100_only[job]/baseline_plus_only[job], 2))
        v100_feedback.append(round(v100_only[job]/feedback_only[job], 2))
        v100_scheme.append(round(v100_only[job]/scheme_only[job], 2))
#        v100_epoch_boundary.append(round(v100_only[job]/epoch_boundary_only[job], 2))
        v100_start_on_both.append(round(v100_only[job]/start_on_both_only[job], 2))
        v100_no_safeguard.append(round(v100_only[job]/no_safeguard_only[job], 2))
        v100_no_threshold.append(round(v100_only[job]/no_threshold_only[job], 2))
        v100_predict_error.append(round(v100_only[job]/predict_error_only[job], 2))

        k80_baseline.append(round(k80_only[job]/baseline_only[job], 2))   
        k80_baseline_plus.append(round(k80_only[job]/baseline_plus_only[job], 2))
        k80_feedback.append(round(k80_only[job]/feedback_only[job], 2))
        k80_scheme.append(round(k80_only[job]/scheme_only[job], 2))
#        k80_epoch_boundary.append(round(k80_only[job]/epoch_boundary_only[job], 2))
        k80_start_on_both.append(round(k80_only[job]/start_on_both_only[job], 2))
        k80_no_safeguard.append(round(k80_only[job]/no_safeguard_only[job], 2))
        k80_no_threshold.append(round(k80_only[job]/no_threshold_only[job], 2))
        k80_predict_error.append(round(k80_only[job]/predict_error_only[job], 2))

    elif i < 100:
        joob = str(i+1-50)
        v100_joob = v100_only[joob]
        k80_joob = k80_only[joob]
        v100_baseline.append(round(v100_joob/baseline_only[job], 2))   
        v100_baseline_plus.append(round(v100_joob/baseline_plus_only[job], 2))
        v100_feedback.append(round(v100_joob/feedback_only[job], 2))
        v100_scheme.append(round(v100_joob/scheme_only[job], 2))
#        v100_epoch_boundary.append(round(v100_joob/epoch_boundary_only[job], 2))
        v100_start_on_both.append(round(v100_joob/start_on_both_only[job], 2))
        v100_no_safeguard.append(round(v100_joob/no_safeguard_only[job], 2))
        v100_no_threshold.append(round(v100_joob/no_threshold_only[job], 2))
        v100_predict_error.append(round(v100_joob/predict_error_only[job], 2))

        k80_baseline.append(round(k80_joob/baseline_only[job], 2))   
        k80_baseline_plus.append(round(k80_joob/baseline_plus_only[job], 2))
        k80_feedback.append(round(k80_joob/feedback_only[job], 2))
        k80_scheme.append(round(k80_joob/scheme_only[job], 2))
#        k80_epoch_boundary.append(round(k80_joob/epoch_boundary_only[job], 2))
        k80_start_on_both.append(round(k80_joob/start_on_both_only[job], 2))
        k80_no_safeguard.append(round(k80_joob/no_safeguard_only[job], 2))
        k80_no_threshold.append(round(k80_joob/no_threshold_only[job], 2))
        k80_predict_error.append(round(k80_joob/predict_error_only[job], 2))

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
cols = zip(norm_feedback)
with open('norm_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_scheme)
with open('norm_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
##cols = zip(norm_epoch_boundary)
##with open('norm_JCT/epoch_boundary_jct.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
cols = zip(norm_start_on_both)
with open('norm_JCT/start_on_both_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_no_safeguard)
with open('norm_JCT/no_safeguard_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_no_threshold)
with open('norm_JCT/no_threshold_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(norm_predict_error)
with open('norm_JCT/predict_error_jct.csv', 'w') as f:
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
cols = zip(v100_feedback)
with open('v100_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_scheme)
with open('v100_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
##cols = zip(v100_epoch_boundary)
##with open('v100_JCT/epoch_boundary_jct.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
cols = zip(v100_start_on_both)
with open('v100_JCT/start_on_both_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_no_safeguard)
with open('v100_JCT/no_safeguard_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_no_threshold)
with open('v100_JCT/no_threshold_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(v100_predict_error)
with open('v100_JCT/predict_error_jct.csv', 'w') as f:
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
cols = zip(k80_feedback)
with open('k80_JCT/feedback_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_scheme)
with open('k80_JCT/scheme_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
##cols = zip(k80_epoch_boundary)
##with open('k80_JCT/epoch_boundary_jct.csv', 'w') as f:
##    writer = csv.writer(f)
##    for col in cols:
##        writer.writerow(col)
cols = zip(k80_start_on_both)
with open('k80_JCT/start_on_both_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_no_safeguard)
with open('k80_JCT/no_safeguard_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_no_threshold)
with open('k80_JCT/no_threshold_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(k80_predict_error)
with open('k80_JCT/predict_error_jct.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)


