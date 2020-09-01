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

testcase = sys.argv[1] # K80_vgg19_32
print(testcase)
base_dir = '/scratch/li.baol/GPU_pwr_meas/tensorflow/round1/'
log_dir = base_dir + testcase + '_*/' # /scratch/li.baol/GPU_pwr_meas/pytorch/K80_vgg19_32_*/
dirs = glob.glob(log_dir)
dirs.sort()
pwr_all = []
avg_all = []

for tc in dirs:
    model = tc.split('/')[5+1]
    files = glob.glob(tc + "sample*.csv")
    files.sort()
    avg_pwr = [0] * (len(files) + 1)
    
    for fil in files:
        file_path = fil
        minute = int(fil.split('/')[6+1].split('_')[1].split('.')[0])
        try: # in case the file is empty
            data = pandas.read_csv(file_path)
            pwr = data[data.columns[2]].tolist()
            
            pwr_array = np.asarray(pwr)
            if (len(pwr) == 0):
                avg_pwr[minute] = 0
            else:
                avg_pwr[minute] = np.average(pwr_array)
        except pandas.errors.EmptyDataError:
            avg_pwr[minute] = 0
            pass
    pwr_all.append(avg_pwr)
    avg_pwr_filter = [i for i in avg_pwr if i > 10] # remove power measurements below 10W
    avg_all.append(sum(avg_pwr_filter) / len(avg_pwr_filter))


#------------- plot ---------------#

width = 0.1

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,5))
fig.suptitle(testcase +  " GPU power (W) during training epochs")
for i in range(len(pwr_all)):    
    x = np.arange(len(pwr_all[i]))
    axs.scatter(x, pwr_all[i], label = str(i))

axs.set_xlabel('# of sample with 10s interval')
axs.set_ylabel('power(W)')
#axs.set_yticks(minor=True)
axs.get_yaxis().set_minor_locator(MultipleLocator(5))
axs.legend()

axs.grid(which='both', axis='y', linestyle=':', color='black')
pwr = int(sum(avg_all) / len(avg_all))
plt.savefig(base_dir + "png/" + testcase + '_pwr' + str(pwr) + ".png")

df = pandas.DataFrame(avg_all, columns=["power(W)"])
df.to_csv(base_dir + 'csv/' + testcase + '.csv', index=False)
