import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

#with open('../final/final5/logs_50/final5_50_JCT.json', 'r') as fp:
#    K80_time = json.load(fp)

with open('dash/logs_sc_40/dash_sc_40mig_JCT.json', 'r') as fp:
    K80_time = json.load(fp)


#print('K80 only JCT')
for i in range(50):
    job = str(i+1)
    print(K80_time[job])
#print('V100 only JCT')
#for i in range(50):
#    job = str(i+1)
#    print(V100_time[job])


