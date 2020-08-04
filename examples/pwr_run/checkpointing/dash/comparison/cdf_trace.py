import json
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

with open('v100_only_JCT.json', 'r') as fp:
    trace_dict = json.load(fp)

trace_list = list(trace_dict.values())

CY = np.cumsum(trace_list)

plot(CY)

show()

