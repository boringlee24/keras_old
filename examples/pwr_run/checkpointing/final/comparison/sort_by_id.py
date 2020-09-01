import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import operator

with open('v100_only_JCT.json', 'r') as fp:
    V100_time = json.load(fp)
with open('k80_only_JCT.json', 'r') as fp:
    K80_time = json.load(fp)


print('K80 only JCT')
for i in range(50):
    job = str(i+1)
    print(K80_time[job])
print('V100 only JCT')
for i in range(50):
    job = str(i+1)
    print(V100_time[job])


