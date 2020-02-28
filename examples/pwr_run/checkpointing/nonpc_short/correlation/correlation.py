import pandas
import pdb
from datetime import datetime
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import sys
from matplotlib.ticker import MultipleLocator
from scipy.stats import pearsonr, spearmanr
import json

with open('speedup.json') as f:
    speedup_dict = json.load(f)
with open('feedback.json') as f:
    feedback_dict = json.load(f)
with open('max_pwr.json') as f:
    max_pwr_dict = json.load(f)
with open('timed_pwr.json') as f:
    timed_pwr_dict = json.load(f)
with open('final1.json') as f:
    final1_dict = json.load(f)

speedup1 = []
feedback = []
speedup2 = []
max_pwr = []
speedup3 = []
timed_pwr = []
speedup4 = []
final1 = []
for i in range(50):
    job = str(i+1)
    if feedback_dict[job] < 1:
        speedup1.append(speedup_dict[job])
        feedback.append(feedback_dict[job])
    job_name = 'job'+job
    if max_pwr_dict[job_name] > 0:
        max_pwr.append(max_pwr_dict[job_name])
        speedup2.append(speedup_dict[job])
    if timed_pwr_dict[job_name] > 0:
        timed_pwr.append(timed_pwr_dict[job_name])
        speedup3.append(speedup_dict[job])
    if final1_dict[job] < 1:
        speedup4.append(speedup_dict[job])
        final1.append(final1_dict[job])

feedback_filtered = []
speedup_filtered = []
for i in range(50):
    job = str(i+1)
    if feedback_dict[job] < 1:
        speedup_filtered.append(speedup_dict[job])
        feedback_filtered.append(feedback_dict[job])

print('feedback to speedup')
pcorr, pp = pearsonr(feedback, speedup1)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
pcorr, pp = spearmanr(feedback, speedup1)
print('Spearman correlation: %.3f, p value %.3f' % (pcorr, pp))

print('feedback_filtered to speedup_filtered')
pcorr, pp = pearsonr(feedback_filtered, speedup_filtered)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
pcorr, pp = spearmanr(feedback_filtered, speedup_filtered)
print('Spearman correlation: %.3f, p value %.3f' % (pcorr, pp))

print('max_pwr to speedup')
pcorr, pp = pearsonr(max_pwr, speedup2)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
pcorr, pp = spearmanr(max_pwr, speedup2)
print('Spearman correlation: %.3f, p value %.3f' % (pcorr, pp))

print('timed_pwr to speedup')
pcorr, pp = pearsonr(timed_pwr, speedup3)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
pcorr, pp = spearmanr(timed_pwr, speedup3)
print('Spearman correlation: %.3f, p value %.3f' % (pcorr, pp))

print('final1 to speedup')
pcorr, pp = pearsonr(final1, speedup4)
print('Pearsons correlation: %.3f, p value %.3f' % (pcorr, pp))
pcorr, pp = spearmanr(final1, speedup4)
print('Spearman correlation: %.3f, p value %.3f' % (pcorr, pp))

