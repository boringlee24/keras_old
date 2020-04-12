import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('../feedback_inverse/logs/feedback_inverse_overhead.json', 'r') as fp:
    feedback_overhead = json.load(fp)
with open('../final4_new/logs/final4_new_overhead.json', 'r') as fp:
    scheme_overhead = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_overhead.json', 'r') as fp:
    no_safeguard_overhead = json.load(fp)
with open('../no_threshold/logs/no_threshold_overhead.json', 'r') as fp:
    no_threshold_overhead = json.load(fp)

with open('../feedback_inverse/logs/feedback_inverse_K80_time.json', 'r') as fp:
    feedback_K80_time = json.load(fp)
with open('../final4_new/logs/final4_new_K80_time.json', 'r') as fp:
    scheme_K80_time = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_K80_time.json', 'r') as fp:
    no_safeguard_K80_time = json.load(fp)
with open('../no_threshold/logs/no_threshold_K80_time.json', 'r') as fp:
    no_threshold_K80_time = json.load(fp)

with open('../feedback_inverse/logs/feedback_inverse_V100_time.json', 'r') as fp:
    feedback_V100_time = json.load(fp)
with open('../final4_new/logs/final4_new_V100_time.json', 'r') as fp:
    scheme_V100_time = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_V100_time.json', 'r') as fp:
    no_safeguard_V100_time = json.load(fp)
with open('../no_threshold/logs/no_threshold_V100_time.json', 'r') as fp:
    no_threshold_V100_time = json.load(fp)

with open('../feedback_inverse/logs/feedback_inverse_k80_1st.json', 'r') as fp:
    feedback_k80_1st = json.load(fp)
with open('../final4_new/logs/final4_new_k80_1st.json', 'r') as fp:
    scheme_k80_1st = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_k80_1st.json', 'r') as fp:
    no_safeguard_k80_1st = json.load(fp)
with open('../no_threshold/logs/no_threshold_k80_1st.json', 'r') as fp:
    no_threshold_k80_1st = json.load(fp)

with open('../feedback_inverse/logs/feedback_inverse_v100_1st.json', 'r') as fp:
    feedback_v100_1st = json.load(fp)
with open('../final4_new/logs/final4_new_v100_1st.json', 'r') as fp:
    scheme_v100_1st = json.load(fp)
with open('../no_safeguard/logs/no_safeguard_v100_1st.json', 'r') as fp:
    no_safeguard_v100_1st = json.load(fp)
with open('../no_threshold/logs/no_threshold_v100_1st.json', 'r') as fp:
    no_threshold_v100_1st = json.load(fp)

print('feedback old average', str(feedback_overhead['average']))
print('scheme old average', str(scheme_overhead['average']))
print('no_threshold old average', str(no_threshold_overhead['average']))
print('no_safeguard old average', str(no_safeguard_overhead['average']))

with open('old_overhead/feedback_K80_time.json', 'w') as fp:
    json.dump(feedback_K80_time, fp, indent=4)
with open('old_overhead/feedback_V100_time.json', 'w') as fp:
    json.dump(feedback_V100_time, fp, indent=4)
with open('old_overhead/feedback_overhead.json', 'w') as fp:
    json.dump(feedback_overhead, fp, indent=4)

with open('old_overhead/scheme_K80_time.json', 'w') as fp:
    json.dump(scheme_K80_time, fp, indent=4)
with open('old_overhead/scheme_V100_time.json', 'w') as fp:
    json.dump(scheme_V100_time, fp, indent=4)
with open('old_overhead/scheme_overhead.json', 'w') as fp:
    json.dump(scheme_overhead, fp, indent=4)

with open('old_overhead/no_threshold_K80_time.json', 'w') as fp:
    json.dump(no_threshold_K80_time, fp, indent=4)
with open('old_overhead/no_threshold_V100_time.json', 'w') as fp:
    json.dump(no_threshold_V100_time, fp, indent=4)
with open('old_overhead/no_threshold_overhead.json', 'w') as fp:
    json.dump(no_threshold_overhead, fp, indent=4)

with open('old_overhead/no_safeguard_K80_time.json', 'w') as fp:
    json.dump(no_safeguard_K80_time, fp, indent=4)
with open('old_overhead/no_safeguard_V100_time.json', 'w') as fp:
    json.dump(no_safeguard_V100_time, fp, indent=4)
with open('old_overhead/no_safeguard_overhead.json', 'w') as fp:
    json.dump(no_safeguard_overhead, fp, indent=4)


with open('v100_time.json', 'r') as fp:
    v100_epoch = json.load(fp)
with open('k80_time.json', 'r') as fp:
    k80_epoch = json.load(fp)

k80 = []
v100 = []

with open('../feedback_inverse/logs/feedback_inverse_birthplace.json', 'r') as fp:
    feedback_birthplace = json.load(fp)

for i in range(100):
    job = str(i+1)
    if i < 50:
        k80_epoch_time = k80_epoch[job]
        v100_epoch_time = v100_epoch[job]
    elif i < 100:
        joob = str(i+1-50)
        k80_epoch_time = k80_epoch[joob]
        v100_epoch_time = v100_epoch[joob]

    ######### special case, start on both K80 and V100 #############

    k80_1st = feedback_k80_1st[job]
    v100_1st = feedback_v100_1st[job]

    if 'c' in feedback_birthplace[job]: # meaning job started on K80
        if len(v100_1st) == 0:
            v100_waste = 0
        else:
            v100_waste = np.sum(v100_1st) - v100_epoch_time * len(v100_1st)

        if len(k80_1st) == 1:
            k80_waste = 0
        elif len(k80_1st) > 1:
            k80_1st.pop(0)
            k80_waste = np.sum(k80_1st) - k80_epoch_time * len(k80_1st)
        elif len(k80_1st) == 0:
            k80_waste = 0
            print('error, uncollected k80 1st epoch time')


    elif 'd' in feedback_birthplace[job]: # meaning job started on V100
        if len(k80_1st) == 0:
            k80_waste = 0
        else:
            k80_waste = np.sum(k80_1st) - k80_epoch_time * len(k80_1st)

        if len(v100_1st) == 1:
            v100_waste = 0
        elif len(v100_1st) > 1:
            v100_1st.pop(0)
            v100_waste = np.sum(v100_1st) - v100_epoch_time * len(v100_1st)
        elif len(v100_1st) == 0:
            v100_waste = 0
            print('error, uncollected v100 1st epoch time')

    feedback_K80_time[job] -= k80_waste
    feedback_V100_time[job] -= v100_waste
    feedback_overhead[job] += v100_waste + k80_waste 
    
    ##############################################################
    
    k80_1st = scheme_k80_1st[job]
    if len(k80_1st) == 0:
        k80_waste = 0
    else:
        k80_waste = np.sum(k80_1st) - k80_epoch_time * len(k80_1st)
    scheme_K80_time[job] -= k80_waste
    v100_1st = scheme_v100_1st[job]
    if len(v100_1st) == 1:
        v100_waste = 0
    elif len(v100_1st) > 1:
        v100_1st.pop(0)
        v100_waste = np.sum(v100_1st) - v100_epoch_time * len(v100_1st)
    scheme_V100_time[job] -= v100_waste
    scheme_overhead[job] += v100_waste + k80_waste 

    k80_1st = no_threshold_k80_1st[job]
    if len(k80_1st) == 0:
        k80_waste = 0
    else:
        k80_waste = np.sum(k80_1st) - k80_epoch_time * len(k80_1st)
    no_threshold_K80_time[job] -= k80_waste
    v100_1st = no_threshold_v100_1st[job]
    if len(v100_1st) == 1:
        v100_waste = 0
    elif len(v100_1st) > 1:
        v100_1st.pop(0)
        v100_waste = np.sum(v100_1st) - v100_epoch_time * len(v100_1st)
    no_threshold_V100_time[job] -= v100_waste
    no_threshold_overhead[job] += v100_waste + k80_waste 

    k80_1st = no_safeguard_k80_1st[job]
    if len(k80_1st) == 0:
        k80_waste = 0
    else:
        k80_waste = np.sum(k80_1st) - k80_epoch_time * len(k80_1st)
    no_safeguard_K80_time[job] -= k80_waste
    v100_1st = no_safeguard_v100_1st[job]
    if len(v100_1st) == 1:
        v100_waste = 0
    elif len(v100_1st) > 1:
        v100_1st.pop(0)
        v100_waste = np.sum(v100_1st) - v100_epoch_time * len(v100_1st)
    no_safeguard_V100_time[job] -= v100_waste
    no_safeguard_overhead[job] += v100_waste + k80_waste

# remove average number
feedback_overhead.pop('average')
scheme_overhead.pop('average')
no_threshold_overhead.pop('average')
no_safeguard_overhead.pop('average')

average = np.mean(list(feedback_overhead.values()))
print('new feedback average', str(average))
average = np.mean(list(scheme_overhead.values()))
print('new scheme average', str(average))
average = np.mean(list(no_threshold_overhead.values()))
print('new no_threshold average', str(average))
average = np.mean(list(no_safeguard_overhead.values()))
print('new no_safeguard average', str(average))

with open('new_overhead/feedback_K80_time.json', 'w') as fp:
    json.dump(feedback_K80_time, fp, indent=4)
with open('new_overhead/feedback_V100_time.json', 'w') as fp:
    json.dump(feedback_V100_time, fp, indent=4)
with open('new_overhead/feedback_overhead.json', 'w') as fp:
    json.dump(feedback_overhead, fp, indent=4)

with open('new_overhead/scheme_K80_time.json', 'w') as fp:
    json.dump(scheme_K80_time, fp, indent=4)
with open('new_overhead/scheme_V100_time.json', 'w') as fp:
    json.dump(scheme_V100_time, fp, indent=4)
with open('new_overhead/scheme_overhead.json', 'w') as fp:
    json.dump(scheme_overhead, fp, indent=4)

with open('new_overhead/no_threshold_K80_time.json', 'w') as fp:
    json.dump(no_threshold_K80_time, fp, indent=4)
with open('new_overhead/no_threshold_V100_time.json', 'w') as fp:
    json.dump(no_threshold_V100_time, fp, indent=4)
with open('new_overhead/no_threshold_overhead.json', 'w') as fp:
    json.dump(no_threshold_overhead, fp, indent=4)

with open('new_overhead/no_safeguard_K80_time.json', 'w') as fp:
    json.dump(no_safeguard_K80_time, fp, indent=4)
with open('new_overhead/no_safeguard_V100_time.json', 'w') as fp:
    json.dump(no_safeguard_V100_time, fp, indent=4)
with open('new_overhead/no_safeguard_overhead.json', 'w') as fp:
    json.dump(no_safeguard_overhead, fp, indent=4)

