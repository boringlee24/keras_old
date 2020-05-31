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
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
from sklearn.model_selection import KFold

log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/regression/pwr/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_pwr = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6+1+1].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    pwr = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_pwr:
        all_pwr[model][gpu] = pwr
    else:
        all_pwr[model] = {gpu: pwr}

log_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/regression/util/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_util = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6+1+1].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    util = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_util:
        all_util[model][gpu] = util
    else:
        all_util[model] = {gpu: util}

log_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/round1/csv/*'
dirs = glob.glob(log_dir)
dirs.sort()
# store everything in a dict
all_time = {} # {densenet121_32:{K80:a, K100:b}...}

for tc in dirs:
    test = tc.split('/')[6+1].split('.')[0]
    gpu = test.split('_')[0]
    model = test.replace(gpu + '_', '')

    # read tc.csv into a list
    data = pandas.read_csv(tc)
    time = np.asarray(data[data.columns[0]].tolist())
    
    if model in all_time:
        all_time[model][gpu] = time
    else:
        all_time[model] = {gpu: time}

# Now plot V100 power save ratio (%) vs K80 power(W)

x1_data = [] # power
x2_data = [] # speed
x3_data = [] # utilization
y_data = []

for key in all_pwr:
#    if ('mnasnet' not in key and 'mobilenet' not in key):
    for i in all_pwr[key]['V100'].tolist(): # power
        x1_data.append(i)
    for i in (1 / all_time[key]['V100']).tolist(): # speed
    #for i in (all_time[key]['V100']).tolist():
        x2_data.append(i)
    for i in (all_util[key]['V100']).tolist(): # utilization
        x3_data.append(i)
    for i in ((all_time[key]['K80'] - all_time[key]['V100']) / all_time[key]['K80'] * 100).tolist(): # speed up  
        y_data.append(i)

x1_norm = [(i - min(x1_data)) / (max(x1_data) - min(x1_data)) for i in x1_data]
x2_norm = [(i - min(x2_data)) / (max(x2_data) - min(x2_data)) for i in x2_data]
x3_norm = [(i - min(x3_data)) / (max(x3_data) - min(x3_data)) for i in x3_data]

# create training data
x_data = []
for i in range(len(x1_norm)):
    x_data.append([x1_norm[i], x2_norm[i], x3_norm[i]])
#    x_data.append([x1_data[i], x3_data[i]])

kf = KFold(n_splits=5, shuffle=True)

x_train_group = []
y_train_group = []
x_test_group = []
y_test_group = []
for train_index, test_index in kf.split(x_data):
#    print('train index:', train_index, 'test index:', test_index)
    x_train = [x_data[i] for i in train_index]
    y_train = [y_data[i] for i in train_index]
    x_test = [x_data[i] for i in test_index]
    y_test = [y_data[i] for i in test_index]
    x_train_group.append(x_train)
    y_train_group.append(y_train)
    x_test_group.append(x_test)
    y_test_group.append(y_test)
 
rmse_val = []
for K in range(20):
    K += 1
    model = neighbors.KNeighborsRegressor(n_neighbors = K, weights='distance')
    err_list = []
    for i in range(len(x_train_group)):
        x_train = x_train_group[i]
        y_train = y_train_group[i]
        x_test = x_test_group[i]
        y_test = y_test_group[i]

        model.fit(x_train, y_train)
        pred = model.predict(x_test) #make prediction on test set
        #err = sqrt(mean_squared_error(y_test, pred)) #calculate rmse
        err = np.mean(abs(y_test - pred) / y_test * 100)
        err_list.append(err)
    rmse_val.append(np.mean(err)) #store rmse values
    print('RMSE value for k= ' , K , 'is:', err)
print('RMSE min= ', np.min(rmse_val))




