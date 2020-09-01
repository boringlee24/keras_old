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

tc_list = ['densenet121_32', 'densenet121_64', 'densenet121_128', 'densenet121_256', 'densenet169_32', 'densenet201_32', 
'mnasnet_32', 'mnasnet_64', 'mnasnet_128', 'mnasnet_256', 
'mobilenet_32', 'mobilenet_64', 'mobilenet_128', 'mobilenet_256', 
'resnet50_32', 'resnet50_64', 'resnet50_128', 'resnet50_256', 'resnet101_32', 'resnet152_32',
'vgg16_32', 'vgg16_64', 'vgg16_128', 'vgg16_256', 'vgg19_32']

vc_pct = [] # variational coefficient in %
spread = []

for testcase in tc_list:
    print(testcase)
    
    pwr_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/csv/K80_' + testcase + '.csv'  # /scratch/li.baol/GPU_pwr_meas/tensorflow/K80_vgg19_32_*/
    time_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/csv/K80_' + testcase + '.csv'  # /scratch/li.baol/GPU_pwr_meas/tensorflow/K80_vgg19_32_*/

    power = pandas.read_csv(pwr_dir)
    power_array = np.asarray(power[power.columns[0]].tolist())

    time = pandas.read_csv(time_dir)
    time_array = np.asarray(time[time.columns[0]].tolist())

    energy_array = power_array * time_array

    mean = np.mean(energy_array)
    stdev = np.std(energy_array)
    vc_pct.append(stdev / mean * 100)
    spread.append((np.amax(energy_array) - np.amin(energy_array)) / np.amin(energy_array))

#------------- plot variation coefficient ---------------#

width = 0.55

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("GPU energy variation coefficient (%) during training")
fig.subplots_adjust(bottom=0.2)

x = np.arange(len(tc_list))
axs.bar(x, vc_pct, width)
axs.set_xticks(x)
axs.set_xticklabels(tc_list, rotation=45, ha="right")
axs.set_xlabel('test cases')
axs.set_ylabel('variation coefficient (%)')
#axs.set_yticks(minor=True)
axs.get_yaxis().set_minor_locator(MultipleLocator(1))
#axs.legend()

axs.grid(which='both', axis='y', linestyle=':', color='black')
plt.savefig("./energy_variation.png")

#------------- plot spread ---------------#

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("GPU energy spread during training")
fig.subplots_adjust(bottom=0.2)

x = np.arange(len(tc_list))
axs.bar(x, spread, width)
axs.set_xticks(x)
axs.set_xticklabels(tc_list, rotation=45, ha="right")
axs.set_xlabel('test cases')
axs.set_ylabel('spread')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(1))
#axs.legend()

axs.grid(which='both', axis='y', linestyle=':', color='black')
#plt.show()
plt.savefig("./energy_spread.png")
