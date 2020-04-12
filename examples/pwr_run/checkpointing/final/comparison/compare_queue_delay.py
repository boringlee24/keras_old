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
with open('../high_overhead/logs/high_overhead_queue_delay.json', 'r') as fp:
    high_overhead_only = json.load(fp)

baseline = []
baseline_plus = []
feedback = []
scheme = []
#epoch_boundary = []
start_on_both = []
no_safeguard = []
no_threshold = []
predict_error = []
high_overhead = []

for i in range(len(baseline_only)-1):
    job = str(i+1)
    baseline.append(baseline_only[job])
    baseline_plus.append(baseline_plus_only[job])
    feedback.append(feedback_only[job])
    scheme.append(scheme_only[job])
#    epoch_boundary.append(epoch_boundary_only[job])
    start_on_both.append(start_on_both_only[job])
    no_safeguard.append(no_safeguard_only[job])
    no_threshold.append(no_threshold_only[job])
    predict_error.append(predict_error_only[job])
    high_overhead.append(high_overhead_only[job])

baseline = np.asarray(baseline)
baseline_plus = np.asarray(baseline_plus)
feedback = np.asarray(feedback)
scheme = np.asarray(scheme)
#epoch_boundary = np.asarray(epoch_boundary)
start_on_both = np.asarray(start_on_both)
no_safeguard = np.asarray(no_safeguard)
no_threshold = np.asarray(no_threshold)
predict_error = np.asarray(predict_error)
high_overhead = np.asarray(high_overhead)

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

