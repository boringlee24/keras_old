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

tc_list1 = ['densenet121_32', 'densenet169_32', 'densenet201_32']
tc_list2 = ['resnet50_32', 'resnet101_32', 'resnet152_32']
tc_list3 = ['vgg16_32', 'vgg19_32']
rounds = ['round1', 'round2']

vc_pct = {'round1': [], 'round2': []} # variational coefficient in %
spread = {'round1': [], 'round2': []}

for rnd in rounds:
    for testcase in tc_list3:
        print(testcase)
        
        pwr_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/' + rnd + '/csv/K80_' + testcase + '.csv'
        time_dir = '/scratch/li.baol/GPU_time_meas/tensorflow/' + rnd + '/csv/K80_' + testcase + '.csv'  
    
        power = pandas.read_csv(pwr_dir)
        power_array = np.asarray(power[power.columns[0]].tolist())
    
        time = pandas.read_csv(time_dir)
        time_array = np.asarray(time[time.columns[0]].tolist())
    
        energy_array = power_array * time_array
    
        mean = np.mean(energy_array)
        stdev = np.std(energy_array)
        vc_pct[rnd].append(stdev / mean * 100)
        spread[rnd].append((np.amax(energy_array) - np.amin(energy_array)) / np.amin(energy_array))

#------------- plot variation coefficient ---------------#

width = 0.35

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("GPU energy variation coefficient (%) during training")
fig.subplots_adjust(bottom=0.2)

x = np.arange(len(tc_list3))
axs.bar(x - width/2, vc_pct['round1'], width, label='round1')
axs.bar(x + width/2, vc_pct['round2'], width, label='round2')
axs.set_xticks(x)
axs.set_xticklabels(tc_list3, rotation=45, ha="right")
axs.set_xlabel('test cases')
axs.set_ylabel('variation coefficient (%)')
#axs.set_yticks(minor=True)
axs.get_yaxis().set_minor_locator(MultipleLocator(1))
axs.legend()

axs.grid(which='both', axis='y', linestyle=':', color='black')
plt.savefig("layers/vgg_layers_variation.png")

#------------- plot spread ---------------#

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle("GPU energy spread during training")
fig.subplots_adjust(bottom=0.2)

x = np.arange(len(tc_list3))
axs.bar(x - width/2, spread['round1'], width, label='round1')
axs.bar(x + width/2, spread['round2'], width, label='round2')
axs.set_xticks(x)
axs.set_xticklabels(tc_list3, rotation=45, ha="right")
axs.set_xlabel('test cases')
axs.set_ylabel('spread')
#axs.set_yticks(minor=True)
#axs.get_yaxis().set_minor_locator(MultipleLocator(1))
axs.legend()

axs.grid(which='both', axis='y', linestyle=':', color='black')
#plt.show()
plt.savefig("layers/vgg_layers_spread.png")
