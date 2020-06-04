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
with open('../feedback_fair/logs/feedback_fair_queue_delay.json', 'r') as fp:
    feedback_new_only = json.load(fp)
with open('../feedback_inverse/logs/feedback_inverse_queue_delay.json', 'r') as fp:
    feedback_only = json.load(fp)
with open('../final4_new/logs/final4_new_queue_delay.json', 'r') as fp:
    scheme_only = json.load(fp)
#with open('../final4_new2/logs/final4_new2_queue_delay.json', 'r') as fp:
#    epoch_boundary_only = json.load(fp)
with open('../final5/logs/final5_queue_delay.json', 'r') as fp:
    start_on_both_only = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_queue_delay.json', 'r') as fp:
    no_safeguard_only = json.load(fp)
with open('../no_threshold/logs/no_threshold_queue_delay.json', 'r') as fp:
    no_threshold_only = json.load(fp)
with open('../predict_error/logs/predict_error_queue_delay.json', 'r') as fp:
    predict_error_only = json.load(fp)
with open('../high_overhead_2/logs/high_overhead_queue_delay.json', 'r') as fp:
    high_overhead_only = json.load(fp)
with open('../final4_new/sensitivity_logs/16_4/final4_new_queue_delay.json', 'r') as fp:
    scheme_16p4_only = json.load(fp)
with open('../final4_new/sensitivity_logs/16_2/final4_new_queue_delay.json', 'r') as fp:
    scheme_16p2_only = json.load(fp)
with open('../final4_new/sensitivity_logs/12_8/final4_new_queue_delay.json', 'r') as fp:
    scheme_12p8_only = json.load(fp)
with open('../final4_new/sensitivity_logs/12_4/final4_new_queue_delay.json', 'r') as fp:
    scheme_12p4_only = json.load(fp)
with open('../high_overhead_2/logs_0.2thres/high_overhead_queue_delay.json', 'r') as fp:
    high_overhead_tuned_only = json.load(fp)
with open('../final5/sensitivity_logs/16_4/final5_queue_delay.json', 'r') as fp:
    scheme_16p4_tuned_only = json.load(fp)
with open('../final5/sensitivity_logs/12_4/final5_queue_delay.json', 'r') as fp:
    scheme_12p4_tuned_only = json.load(fp)

baseline = []
baseline_plus = []
feedback = []
feedback_new = []
scheme = []
#epoch_boundary = []
start_on_both = []
no_safeguard = []
no_threshold = []
predict_error = []
high_overhead = []
scheme_16p4 = []
scheme_16p2 = []
scheme_12p8 = []
scheme_12p4 = []
high_overhead_tuned = []
scheme_16p4_tuned = []
scheme_12p4_tuned = []

for i in range(len(baseline_only)-1):
    job = str(i+1)
    baseline.append(baseline_only[job])
    baseline_plus.append(baseline_plus_only[job])
    feedback.append(feedback_only[job])
    feedback_new.append(feedback_new_only[job])
    scheme.append(scheme_only[job])
#    epoch_boundary.append(epoch_boundary_only[job])
    start_on_both.append(start_on_both_only[job])
    no_safeguard.append(no_safeguard_only[job])
    no_threshold.append(no_threshold_only[job])
    predict_error.append(predict_error_only[job])
    high_overhead.append(high_overhead_only[job])
    scheme_16p4.append(scheme_16p4_only[job])
    scheme_16p2.append(scheme_16p2_only[job])
    scheme_12p8.append(scheme_12p8_only[job])
    scheme_12p4.append(scheme_12p4_only[job])
    high_overhead_tuned.append(high_overhead_tuned_only[job])
    scheme_16p4_tuned.append(scheme_16p4_tuned_only[job])
    scheme_12p4_tuned.append(scheme_12p4_tuned_only[job])

baseline = np.asarray(baseline)
baseline_plus = np.asarray(baseline_plus)
feedback = np.asarray(feedback)
feedback_new = np.asarray(feedback_new)
scheme = np.asarray(scheme)
#epoch_boundary = np.asarray(epoch_boundary)
start_on_both = np.asarray(start_on_both)
no_safeguard = np.asarray(no_safeguard)
no_threshold = np.asarray(no_threshold)
predict_error = np.asarray(predict_error)
high_overhead = np.asarray(high_overhead)
scheme_16p4 = np.asarray(scheme_16p4)
scheme_16p2 = np.asarray(scheme_16p2)
scheme_12p8 = np.asarray(scheme_12p8)
scheme_12p4 = np.asarray(scheme_12p4)
high_overhead_tuned = np.asarray(high_overhead_tuned)
scheme_16p4_tuned = np.asarray(scheme_16p4_tuned)
scheme_12p4_tuned = np.asarray(scheme_12p4_tuned)

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
cols = zip(feedback)
with open('queue_delay/feedback_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(feedback_new)
with open('queue_delay/feedback_new_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme)
with open('queue_delay/scheme_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
#cols = zip(epoch_boundary)
#with open('queue_delay/epoch_boundary_queue_delay.csv', 'w') as f:
#    writer = csv.writer(f)
#    for col in cols:
#        writer.writerow(col)
cols = zip(start_on_both)
with open('queue_delay/start_on_both_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(no_safeguard)
with open('queue_delay/no_safeguard_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(no_threshold)
with open('queue_delay/no_threshold_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(predict_error)
with open('queue_delay/predict_error_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(high_overhead)
with open('queue_delay/high_overhead_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_16p4)
with open('queue_delay/scheme_16p4_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_16p2)
with open('queue_delay/scheme_16p2_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_12p8)
with open('queue_delay/scheme_12p8_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_12p4)
with open('queue_delay/scheme_12p4_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(high_overhead_tuned)
with open('queue_delay/high_overhead_tuned_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_16p4_tuned)
with open('queue_delay/scheme_16p4_tuned_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)
cols = zip(scheme_12p4_tuned)
with open('queue_delay/scheme_12p4_tuned_queue_delay.csv', 'w') as f:
    writer = csv.writer(f)
    for col in cols:
        writer.writerow(col)

