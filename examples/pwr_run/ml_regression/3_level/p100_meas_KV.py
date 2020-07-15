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
    for i in all_pwr[key]['P100'].tolist(): # power
        x1_data.append(i)
    for i in (1 / all_time[key]['P100']).tolist(): # speed
        x2_data.append(i)
    for i in (all_util[key]['P100']).tolist(): # utilization
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

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

with open('x1_data.json', 'w') as outfile:
    json.dump(x1_data, outfile)
with open('x2_data.json', 'w') as outfile:
    json.dump(x2_data, outfile)
with open('x3_data.json', 'w') as outfile:
    json.dump(x3_data, outfile)

with open('y_data.json', 'w') as outfile:
    json.dump(y_data, outfile)
#with open('x_data.json') as f:
#    x_data = json.load(f)
#with open('y_data.json') as f:
#    y_data = json.load(f)
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K, weights='distance')

    model.fit(x_train, y_train)  #fit the model
    pred = model.predict(x_test) #make prediction on test set
#    model.predict(np.array(x_test[0]).reshape((1, -1)))
    err = sqrt(mean_squared_error(y_test, pred)) #calculate rmse
    rmse_val.append(err) #store rmse values
    print('RMSE value for k= ' , K , 'is:', err)

xx_data = []
for i in range(len(x1_norm)):
    xx_data.append([x1_norm[i]])

# now compare with liear regression
x_train, x_test, y_train, y_test = train_test_split(xx_data, y_data, test_size=0.3)
model2 = LinearRegression().fit(x_train, y_train)
pred = model2.predict(x_test) #make prediction on test set
err = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
print('RMSE value for linear regression is ', err)



