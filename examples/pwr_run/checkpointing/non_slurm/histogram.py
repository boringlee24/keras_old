import glob
import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

testcase = 'random2'
JCT_log = './random2/logs/' + testcase + '_JCT.json'
PJCT_log = './random2/logs/' + testcase + '_PJCT.json'

JCT = {}
PJCT = {}
with open(JCT_log, 'r') as fp:
    JCT = json.load(fp)
with open(PJCT_log, 'r') as fp:
    PJCT = json.load(fp)

random2_time = []
for i in range(50):
    job = str(i+1)
    if job in PJCT:
        random2_time.append(PJCT[job] / JCT[job] * 100)
    else:
        random2_time.append(JCT[job] / JCT[job] * 100)

def get_cdf(data):
    """Returns the CDF of the given data.
    
       Args:
           data: A list of numerical values.
           
       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p

x, y = get_cdf(random2_time)
plt.plot(x, y)
plt.ylim(0, 100)
#plt.xlabel('Time (min)')
plt.xlabel('portion of practical complete time in total job runtime (%)')
plt.ylabel('CDF')
plt.grid(alpha=.3, linestyle='--')
plt.title('CDF of the percentage of job practical complete time in total time ')
#random2_time = np.array(random2_time, dtype=np.float)



#hist, bins = np.histogram(random2_time, bins=10, normed=True)
#bin_centers = (bins[1:]+bins[:-1])*0.5
#plt.bar(bin_centers, hist)
plt.savefig('cdf.png')

       
