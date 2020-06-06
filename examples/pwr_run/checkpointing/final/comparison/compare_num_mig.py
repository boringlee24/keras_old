import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator
import pdb

with open('../feedback_v100_only/logs/feedback_fair_num_mig.json', 'r') as fp:
    feedback_only = json.load(fp)
with open('../final4_new/logs/final4_new_num_mig.json', 'r') as fp:
    scheme_only = json.load(fp)

feedback_num = list(feedback_only.values())
scheme_num = list(scheme_only.values())

print('total migrations of feedback:', str(sum(feedback_num)), 'total of scheme:', str(sum(scheme_num)))


