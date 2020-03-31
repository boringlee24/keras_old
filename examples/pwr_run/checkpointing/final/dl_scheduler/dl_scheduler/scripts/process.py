# Import libraries
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as mp
import matplotlib.gridspec as gridspec

# Define Matplotlib font properties
mpl.use("pgf")

preamble = [
    r'\usepackage{fontspec}',
    r'\usepackage{physics}'
]

params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': preamble,
}

mpl.rcParams.update(params)

POLICIES = ['baseline', 'baseline+', 'feedback', 'scheme']#, 'scheme_new']
TITLES = ['BASELINE+', 'FEEDBACK', 'PROPOSED']#, 'PROPOSED_NEW']
COLORS = ['gold', 'tomato', 'navy']#, 'orange']

'''
data = []

# Get the job completion times
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/JCT/' + p + '_jct.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		dat[-1].append(int(line))

	dat[-1] = np.mean(dat[-1])

#for d in range(len(dat)-1, 0, -1):
#	dat[d] = stats.mstats.gmean([dat[0][x]/dat[d][x] for x in range(len(dat[d]))])
#	dat[d] -= 1
#	dat[d] *= 100
#	print(dat)

#data.append([dat[x] for x in range(1, len(dat))])
data.append([(dat[0] - dat[x]) / dat[0] * 100 for x in range(1, len(dat))])

# Get the queue delay times
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/queue_delay/' + p + '_queue_delay.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		dat[-1].append(int(line))

	dat[-1] = np.mean(dat[-1])

data.append([(dat[0] - dat[x]) / dat[0] * 100 for x in range(1, len(dat))])

# Get the throughput
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/throughput/' + p + '_gpu_usage.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		if 'time' in line:
			continue

		line = line.split(',')

		dat[-1].append([int(line[0]), int(line[1]), float(line[2])])

	t = 7208
	if p == 'baseline+':
		t = 7567
	elif p == 'feedback':
		t = 7495
	elif p == 'scheme':
		t = 6416

	for d in dat[-1]:
		if d[0] == t:
			dat[-1] = d[0]/d[2]
			break
		prev = d

data.append([(dat[0] - dat[x]) / dat[0] * 100 for x in range(1, len(dat))])

fig = mp.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.12, right=0.995, top=0.85, bottom=0.035, wspace=0.7)

axl = fig.add_subplot(111, frameon=False)
axl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axl.set_xticks([])
axl.set_yticks([])

gs = fig.add_gridspec(1, 3)

pl = []
ax = []
for i in range(3):
	ax.append(fig.add_subplot(gs[0, i]))

	ax[i].set_axisbelow(True)
	ax[i].grid(color='lightgrey', linestyle=':')

	if i == 0:
		for j in range(len(TITLES)):
			pl.append(ax[i].bar(j, data[i][j], linewidth=1, color=COLORS[j], edgecolor='black'))
		ax[i].set_xticks([])
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
		mp.axhline(y=0, color='black', linewidth=1)
		ax[i].set_ylabel('Mean Job Completion Time\n' + r'(% Improvement Over BASE)', fontsize=14)
	elif i == 1:
		for j in range(len(TITLES)):
			pl.append(ax[i].bar(j, data[i][j], linewidth=1, color=COLORS[j], edgecolor='black'))
		ax[i].set_xticks([])
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
		mp.axhline(y=0, color='black', linewidth=1)
		ax[i].set_ylabel('Mean Queue Delay Time\n' + r'(% Improvement Over BASE)', fontsize=14)
	elif i == 2:
		for j in range(len(TITLES)):
			pl.append(ax[i].bar(j, data[i][j], linewidth=1, color=COLORS[j], edgecolor='black'))
		ax[i].set_xticks([])
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
		mp.axhline(y=0, color='black', linewidth=1)
		ax[i].set_ylabel('System Job Throughput\n' + r'(% Improvement Over BASE)', fontsize=14)

axl.legend(pl, TITLES, bbox_to_anchor=(0., 1.03, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., fontsize=14, edgecolor='black')

mp.savefig('../figures/improvements_over_baseline.pdf')
mp.close()
'''

'''
data = []

# Get the job completion times
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/JCT/' + p + '_jct.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		dat[-1].append(int(line))

for d in range(len(dat)-1, -1, -1):
	dat[d] = sorted([(dat[0][x]-dat[d][x])/dat[0][x]*100.0 for x in range(len(dat[d]))])

data.append(dat[1:])

# Get the queue delay times
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/queue_delay/' + p + '_queue_delay.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		dat[-1].append(int(line))

for d in range(len(dat)-1, -1, -1):
	dat[d] = sorted([(dat[0][x]-dat[d][x])/dat[0][x]*100.0 for x in range(len(dat[d]))])

data.append(dat[1:])

fig = mp.figure(figsize=(6, 3))
fig.subplots_adjust(left=0.17, right=0.975, top=0.85, bottom=0.175, wspace=0.6)

axl = fig.add_subplot(111, frameon=False)
axl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axl.set_xticks([])
axl.set_yticks([])

gs = fig.add_gridspec(1, 2)

pl = []
ax = []
for i in range(2):
	ax.append(fig.add_subplot(gs[0, i]))

	ax[i].set_axisbelow(True)
	ax[i].grid(color='lightgrey', linestyle=':')

	if i == 0:
		for j in range(len(TITLES)):
			p, = ax[i].plot(list(range(1, len(data[i][j])+1)), data[i][j], linewidth=1.5, color=COLORS[j])
			pl.append(p)
		ax[i].set_xlim([1, len(data[i][0])])
		ax[i].set_ylim([-100, 100])
		mp.setp(ax[i].get_xticklabels(), fontsize=14)
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
		mp.axhline(y=0, color='black', linewidth=1)
		ax[i].set_xlabel('Job Id ' + r'(sorted)', fontsize=14)
		ax[i].set_ylabel('Ind. Job Completion Time\n' + r'(% Improve. Over BASE)', fontsize=14)
	elif i == 1:
		for j in range(len(TITLES)):
			p, = ax[i].plot(list(range(1, len(data[i][j])+1)), data[i][j], linewidth=1.5, color=COLORS[j])
			pl.append(p)
		ax[i].set_xlim([1, len(data[i][0])])
		ax[i].set_ylim([-100, 100])
		mp.setp(ax[i].get_xticklabels(), fontsize=14)
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
		mp.axhline(y=0, color='black', linewidth=1)
		ax[i].set_xlabel('Job Id ' + r'(sorted)', fontsize=14)
		ax[i].set_ylabel('Ind. Job Queue Time\n' + r'(% Improve. Over BASE)', fontsize=14)

axl.legend(pl, TITLES, bbox_to_anchor=(0., 1.05, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., fontsize=11, edgecolor='black')

mp.savefig('../figures/individual_job_time_improvements_over_baseline.pdf')
mp.close()
'''

data = []

# Get the job completion times
dat = []
for p in POLICIES:
	dat.append([])
	with open('../data/JCT/' + p + '_jct.csv', 'r') as f:
		lines = f.readlines()

	for line in lines:

		line = line.strip()

		dat[-1].append(int(line))

v100_jct = []
with open('../data/JCT/v100_jct.csv', 'r') as f:
	lines = f.readlines()

for line in lines:

	line = line.strip()

	v100_jct.append(float(line)/3600.0)

dt = [{} for x in range(len(TITLES))]
for x in range(len(v100_jct)):
	d = v100_jct[x]
	b = None
	if d < 0.25:
		b = '0.25'
	elif d <0.5:
		b = '0.5'
	elif d <0.75:
		b = '0.75'
	elif d <1.0:
		b = '1.0'
	else:
		b = r'>1.0'
	if b not in dt[0]:
		dt[0][b] = [0, 0]
		dt[1][b] = [0, 0]
		dt[2][b] = [0, 0]
	dt[0][b][0] += dat[0][x]
	dt[1][b][0] += dat[0][x]
	dt[2][b][0] += dat[0][x]
	dt[0][b][1] += dat[1][x]
	dt[1][b][1] += dat[2][x]
	dt[2][b][1] += dat[3][x]

for p in range(len(dt)):
	data.append([])
	for b in ['0.25', '0.5', '0.75', '1.0', '>1.0']:
		data[-1].append((dt[p][b][0] - dt[p][b][1]) / dt[p][b][0] * 100)

fig = mp.figure(figsize=(7, 3))
fig.subplots_adjust(left=0.135, right=0.995, top=0.85, bottom=0.18, wspace=0.35)

axl = fig.add_subplot(111, frameon=False)
axl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axl.set_xticks([0])
axl.set_yticks([])
mp.setp(axl.get_xticklabels(), fontsize=14, color='white')

gs = fig.add_gridspec(1, 3)

pl = []
ax = []
for i in range(3):
	ax.append(fig.add_subplot(gs[0, i]))

	ax[i].set_axisbelow(True)
	ax[i].grid(color='lightgrey', linestyle=':')

	if i == 0:
		pl.append(ax[i].bar(range(len(data[i])), data[i], linewidth=1, color=COLORS[i], edgecolor='black'))
		ax[i].plot(range(len(data[i])), data[i], linewidth=1, color='red', marker='o', markersize=5, markeredgecolor='black', markeredgewidth=0.8)
		ax[i].axhline(y=0, color='black', linewidth=1)

		ax[i].set_xlim([-0.6, 4.6])
		ax[i].set_xticks([0, 1, 2, 3, 4])
		ax[i].set_xticklabels(['15', '30', '45', '60', '>60'])

		ax[i].set_ylim([-40, 40])

		mp.setp(ax[i].get_xticklabels(), fontsize=14)
		mp.setp(ax[i].get_yticklabels(), fontsize=14)
	
		ax[i].set_ylabel('Mean Job Completion Time\n' + r'(% Improve. Over BASE)', fontsize=14)
	else:
		pl.append(ax[i].bar(range(len(data[i])), data[i], linewidth=1, color=COLORS[i], edgecolor='black'))
		ax[i].plot(range(len(data[i])), data[i], linewidth=1, color='red', marker='o', markersize=5, markeredgecolor='black', markeredgewidth=0.8)
		ax[i].axhline(y=0, color='black', linewidth=1)

		ax[i].set_xlim([-0.6, 4.6])
		ax[i].set_xticks([0, 1, 2, 3, 4])
		ax[i].set_xticklabels(['15', '30', '45', '60', '>60'])

		ax[i].set_ylim([-40, 40])

		mp.setp(ax[i].get_xticklabels(), fontsize=14)
		mp.setp(ax[i].get_yticklabels(), fontsize=14)

axl.set_xlabel('Job Completion Time on V100 ' +  r'$(minutes)$', fontsize=14)
axl.legend(pl, TITLES, bbox_to_anchor=(0., 1.05, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., fontsize=13, edgecolor='black')

mp.savefig('../figures/job_improvement_by_jct_on_V100.pdf')
mp.close()
