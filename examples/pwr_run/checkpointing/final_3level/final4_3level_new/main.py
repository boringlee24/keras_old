import pdb
import time
import os
import subprocess
import re
import random
import json
import numpy as np
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import socket
import argparse
import threading
import _thread
import signal
from datetime import datetime
import csv
from sklearn import neighbors
import gpu_pwr

parser = argparse.ArgumentParser(description='TCP client')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()

with open('job_queue.json', 'r') as fp:
    queue = json.load(fp)
queue_dict = {}
arrival_time = 0 
for item in queue:
    arrival_time += np.random.poisson(30)
    queue_dict[item] = arrival_time
queue_timer = time.time()
queue_delay = {}
for item in queue:
    queue_delay[str(item)] = 0

job_start = {} #{'49': time1, '15': time2...}
JCT = {}
for item in queue:
    JCT[str(item)] = 0
completion = {}
for item in queue:
    completion[str(item)] = 0
overhead = {} # initialize so that every job starts with 0s overhead time
for item in queue:
    overhead[str(item)] = 0
ovhd_start = {} # initialize this to 0 as well
for item in queue:
    ovhd_start[str(item)] = 0
b_start = {} # initialize this to 0 as well
for item in queue:
    b_start[str(item)] = 0
c_start = {} # initialize this to 0 as well
for item in queue:
    c_start[str(item)] = 0
d_start = {} # initialize this to 0 as well
for item in queue:
    d_start[str(item)] = 0

ovhd_a = {} # {1: [10, 12, ...], 2: [xx]} 
for item in queue:
    ovhd_a[str(item)] = []
ovhd_b = {} # {1: [10, 12, ...], 2: [xx]} 
for item in queue:
    ovhd_b[str(item)] = []
ovhd_c = {} # {1: [10, 12, ...], 2: [xx]} 
for item in queue:
    ovhd_c[str(item)] = []
ovhd_d = {} # {1: [10, 12, ...], 2: [xx]} 
for item in queue:
    ovhd_d[str(item)] = []
ovhd_total = {} # {1: [10, 12, ...], 2: [xx]} 
for item in queue:
    ovhd_total[str(item)] = []
k80_1st = {}
for item in queue:
    k80_1st[str(item)] = []
p100_1st = {}
for item in queue:
    p100_1st[str(item)] = []
v100_1st = {}
for item in queue:
    v100_1st[str(item)] = []

num_mig = {} # initialize migration time to 0
for item in queue:
    num_mig[str(item)] = 0
V100_epoch_time = {}
for item in queue:
    V100_epoch_time[str(item)] = 0
K80_epoch_time = {}
for item in queue:
    K80_epoch_time[str(item)] = 0
P100_epoch_time = {}
for item in queue:
    P100_epoch_time[str(item)] = 0

K80_start_time = {}
for item in queue:
    K80_start_time[str(item)] = 0
P100_start_time = {}
for item in queue:
    P100_start_time[str(item)] = 0
V100_start_time = {}
for item in queue:
    V100_start_time[str(item)] = 0

promote_start_time = {}
for item in queue:
    promote_start_time[str(item)] = 0
V100_demote_list = []
P100_demote_list = []

K80_time = {}
for item in queue:
    K80_time[str(item)] = 0
P100_time = {}
for item in queue:
    P100_time[str(item)] = 0
V100_time = {}
for item in queue:
    V100_time[str(item)] = 0
gpu_usage_time = [] # don't initialize this
gpu_usage = []
gpu_usage_completion = []

speedup_dict_V100 = {}
for item in queue:
    speedup_dict_V100[str(item)] = 0 
speedup_dict_P100 = {}
for item in queue:
    speedup_dict_P100[str(item)] = 0 
predict_dict_V100 = {}
for item in queue:
    predict_dict_V100[str(item)] = 0 
predict_dict_P100 = {}
for item in queue:
    predict_dict_P100[str(item)] = 0 

birthplace = {}
for item in queue:
    birthplace[str(item)] = 'none' 

index = 0
all_jobs_started = False

K80_cap = 16
P100_cap = 4
V100_cap = 4
K80_used = 0
P100_used = 0
V100_used = 0

K80_job = {}
for i in range(K80_cap):
    K80_job[str(i)] = 'idle'
P100_job = {}
for i in range(P100_cap):
    P100_job[str(i)] = 'idle'
V100_job = {}
for i in range(V100_cap):
    V100_job[str(i)] = 'idle'
qualified_job = []
step1_job_P100 = []
step1_job_V100 = []
step2_job = []
pc_job = []

K80_node = ['c2179', 'c2183']
P100_node = ['c2189']
V100_node = ['d1015']
host_node = 'c0172'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s
run_log = open('run.log','w')

def K80_LUT(gpu):
    quotient = int(gpu) // 8
    remainder = int(gpu) % 8
    real_node = K80_node[quotient]
    real_gpu = str(remainder)
    return real_node, real_gpu
def P100_LUT(gpu):
    quotient = int(gpu) // 4
    remainder = int(gpu) % 4
    real_node = P100_node[quotient]
    real_gpu = str(remainder)
    return real_node, real_gpu
def V100_LUT(gpu):
    quotient = int(gpu) // 4
    remainder = int(gpu) % 4
    real_node = V100_node[quotient]
    real_gpu = str(remainder)
    return real_node, real_gpu

######################### do a regression fit ########################
with open('v100_meas_data/x1_data.json') as f:
    x1_v100 = json.load(f)
with open('v100_meas_data/x2_data.json') as f:
    x2_v100 = json.load(f)
with open('v100_meas_data/x3_data.json') as f:
    x3_v100 = json.load(f)
x1_norm = [(i - min(x1_v100)) / (max(x1_v100) - min(x1_v100)) for i in x1_v100]
x2_norm = [(i - min(x2_v100)) / (max(x2_v100) - min(x2_v100)) for i in x2_v100]
x3_norm = [(i - min(x3_v100)) / (max(x3_v100) - min(x3_v100)) for i in x3_v100]
# create training data
x_train = []
for i in range(len(x1_norm)):
    x_train.append([x1_norm[i], x2_norm[i], x3_norm[i]])
with open('v100_meas_data/y_data_KV.json') as f:
    y_train_KV = json.load(f)
model_V100_KV = neighbors.KNeighborsRegressor(n_neighbors = 3, weights='distance')
model_V100_KV.fit(x_train, y_train_KV)
with open('v100_meas_data/y_data_KP.json') as f:
    y_train_KP = json.load(f)
model_V100_KP = neighbors.KNeighborsRegressor(n_neighbors = 3, weights='distance')
model_V100_KP.fit(x_train, y_train_KP)

with open('p100_meas_data/x1_data.json') as f:
    x1_p100 = json.load(f)
with open('p100_meas_data/x2_data.json') as f:
    x2_p100 = json.load(f)
with open('p100_meas_data/x3_data.json') as f:
    x3_p100 = json.load(f)
x1_norm = [(i - min(x1_p100)) / (max(x1_p100) - min(x1_p100)) for i in x1_p100]
x2_norm = [(i - min(x2_p100)) / (max(x2_p100) - min(x2_p100)) for i in x2_p100]
x3_norm = [(i - min(x3_p100)) / (max(x3_p100) - min(x3_p100)) for i in x3_p100]
# create training p100
x_train = []
for i in range(len(x1_norm)):
    x_train.append([x1_norm[i], x2_norm[i], x3_norm[i]])
with open('p100_meas_data/y_data_KP.json') as f:
    y_train_KP = json.load(f)
model_P100_KP = neighbors.KNeighborsRegressor(n_neighbors = 3, weights='distance')
model_P100_KP.fit(x_train, y_train_KP)
with open('p100_meas_data/y_data_KV.json') as f:
    y_train_KV = json.load(f)
model_P100_KV = neighbors.KNeighborsRegressor(n_neighbors = 3, weights='distance')
model_P100_KV.fit(x_train, y_train_KV)

####################################################################

def send_signal(node, cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 10000 
    # Connect the socket to the port where the server is listening
    server_address = (node, int(port))

    print('connecting to {} port {}'.format(*server_address), file=run_log, flush=True)
    sock.connect(server_address)

    try:
        # Send data
        message = cmd.encode('utf-8') #b'save 35'  #b'start 35 gpu 6'#b'save 35'
 
        print('sending {!r}'.format(message), file=run_log, flush=True)
        sock.sendall(message)
        while True:
            data = sock.recv(32)
            if 'success' in data.decode('utf-8'):
#                print('received {!r}'.format(data))
                break
            else:
                print('waiting for success signal', file=run_log, flush=True)
                time.sleep(1)
    finally:
        #print('closing socket')
        sock.close()

def max_speedup_promotion_P2V(V100_free, promote_list):
    num_promote = len(promote_list)
    global speedup_dict_V100

    # selectively promote among active V100 jobs and promote list jobs
    V100_pool = promote_list    
    if num_promote <= V100_free: # promote all jobs as well
        return promote_list[:], []
    else: # promote the top 4 jobs            
        pool_dict = {}
        V100_avail = V100_free
        for job in V100_pool:
            if job in speedup_dict_V100:
                pool_dict[job] = speedup_dict_V100[job]        
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:V100_avail] 
        promotion_list = list(set(promote_list).intersection(sorted_pool))                     
        demotion_list = [] 
        return promotion_list, demotion_list

def max_speedup_promotion_V100(V100_free, promote_list, demote_list):
    num_promote = len(promote_list)
    global speedup_dict_V100

    # selectively promote among active V100 jobs and promote list jobs
    V100_pool = list(set(demote_list).union(promote_list))       
    if num_promote <= V100_free: # promote all jobs as well
        return promote_list[:], []
    else: # promote the top 4 jobs            
        pool_dict = {}
        V100_avail = V100_free + len(demote_list)
        for job in V100_pool:
            if job in speedup_dict_V100:
                pool_dict[job] = speedup_dict_V100[job]        
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:V100_avail] 
        promotion_list = list(set(promote_list).intersection(sorted_pool))                     
        demotion_list = list(set(demote_list).difference(sorted_pool))
        if 'idle' in demotion_list:
            demotion_list.remove('idle') # this includes force demotion

        # lazy migration, for every V100 job from high speeup to low speedup and not in sorted_pool, compare it with
        # K80 jobs in sorted_pool, from low speedup to high speedup. If difference within 0.2, replace the K80 job
        # in sorted pool
        for job_demote in sorted(pool_dict, key=pool_dict.get, reverse=True):
            if job_demote in demotion_list:
                for job_promote in sorted(pool_dict, key=pool_dict.get, reverse=False):
                    if job_promote in promotion_list:
                        if speedup_dict_V100[job_promote] - speedup_dict_V100[job_demote] < 0.3:
                            demotion_list.remove(job_demote)
                            promotion_list.remove(job_promote)
                            break
        return promotion_list, demotion_list

def max_speedup_promotion_P100(P100_free, promote_list, demote_list):
    num_promote = len(promote_list)
    global speedup_dict_P100

    # selectively promote among active P100 jobs and promote list jobs
    P100_pool = list(set(demote_list).union(promote_list))       
    if num_promote <= P100_free: # promote all jobs as well
        return promote_list[:], []
    else: # promote the top 4 jobs            
        pool_dict = {}
        P100_avail = P100_free + len(demote_list)
        for job in P100_pool:
            if job in speedup_dict_P100:
                pool_dict[job] = speedup_dict_P100[job]        
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:P100_avail] 
        promotion_list = list(set(promote_list).intersection(sorted_pool))                     
        demotion_list = list(set(demote_list).difference(sorted_pool))
        if 'idle' in demotion_list:
            demotion_list.remove('idle') # this includes force demotion

        # lazy migration, for every P100 job from high speeup to low speedup and not in sorted_pool, compare it with
        # K80 jobs in sorted_pool, from low speedup to high speedup. If difference within 0.2, replace the K80 job
        # in sorted pool
        for job_demote in sorted(pool_dict, key=pool_dict.get, reverse=True):
            if job_demote in demotion_list:
                for job_promote in sorted(pool_dict, key=pool_dict.get, reverse=False):
                    if job_promote in promotion_list:
                        if speedup_dict_P100[job_promote] - speedup_dict_P100[job_demote] < 0.3:
                            demotion_list.remove(job_demote)
                            promotion_list.remove(job_promote)
                            break
        return promotion_list, demotion_list

def min_speedup_demotion_V100(promote_list, demote_list):
    global speedup_dict_V100

    K80_pool = list(set(promote_list).union(demote_list))       
    pool_dict = {}
    for job in K80_pool:
        if job in speedup_dict_V100:
            pool_dict[job] = speedup_dict_V100[job]        
    sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:len(promote_list)] # least speedup jobs
    demotion_list = list(set(demote_list).intersection(sorted_pool))
    promotion_list = list(set(promote_list).difference(sorted_pool))
    # lazy migration, for every V100 job from high speeup to low speedup and not in sorted_pool, compare it with
    # K80 jobs in sorted_pool, from low speedup to high speedup. If difference within 0.2, replace the K80 job
    # in sorted pool
    for job_demote in sorted(pool_dict, key=pool_dict.get, reverse=True):
        if job_demote in demotion_list:
            for job_promote in sorted(pool_dict, key=pool_dict.get, reverse=False):
                if job_promote in promotion_list:
                    if speedup_dict_V100[job_promote] - speedup_dict_V100[job_demote] < 0.15:
                        demotion_list.remove(job_demote)
                        promotion_list.remove(job_promote)
                        break

    return promotion_list, demotion_list

def min_speedup_demotion_P100(promote_list, demote_list): # in this case available jobs have all demoted to idle K80s
    global speedup_dict_P100

    K80_pool = list(set(promote_list).union(demote_list))       
    pool_dict = {}
    for job in K80_pool:
        if job in speedup_dict_P100:
            pool_dict[job] = speedup_dict_P100[job]        
    sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:len(promote_list)] # least speedup jobs
    demotion_list = list(set(demote_list).intersection(sorted_pool))
    promotion_list = list(set(promote_list).difference(sorted_pool))
    # lazy migration, for every V100 job from high speeup to low speedup and not in sorted_pool, compare it with
    # P100 jobs in sorted_pool, from low speedup to high speedup. If difference within 0.2, replace the P100 job
    # in sorted pool
    for job_demote in sorted(pool_dict, key=pool_dict.get, reverse=True):
        if job_demote in demotion_list:
            for job_promote in sorted(pool_dict, key=pool_dict.get, reverse=False):
                if job_promote in promotion_list:
                    if speedup_dict_P100[job_promote] - speedup_dict_P100[job_demote] < 0.15: # use 0.1 for P100 pairs
                        demotion_list.remove(job_demote)
                        promotion_list.remove(job_promote)
                        break

    return promotion_list, demotion_list

def min_speedup_demotion_free(K80_free, P100_demote_list, V100_demote_list, have_to_demote):
    global speedup_dict_P100, speedup_dict_V100

    if have_to_demote >= K80_free:
        if len(P100_demote_list) + len(V100_demote_list) <= K80_free:
            return P100_demote_list[:], V100_demote_list[:] # returns P100 demoted, V100 demoted
        else:
            K80_pool = list(set(P100_demote_list).union(V100_demote_list))
            pool_dict = {}
            for job in K80_pool:
                if job in P100_demote_list:
                    pool_dict[job] = speedup_dict_P100[job]
                elif job in V100_demote_list:
                    pool_dict[job] = speedup_dict_V100[job]
            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_free] # least speedup jobs
            P100_demotion = list(set(P100_demote_list).intersection(sorted_pool))
            V100_demotion = list(set(V100_demote_list).intersection(sorted_pool))
            return P100_demotion, V100_demotion
    else:
        if len(P100_demote_list) + len(V100_demote_list) <= have_to_demote:
            return P100_demote_list[:], V100_demote_list[:] # returns P100 demoted, V100 demoted
        else:
            K80_pool = list(set(P100_demote_list).union(V100_demote_list))
            pool_dict = {}
            for job in K80_pool:
                if job in P100_demote_list:
                    pool_dict[job] = speedup_dict_P100[job]
                elif job in V100_demote_list:
                    pool_dict[job] = speedup_dict_V100[job]
            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:have_to_demote] # least speedup jobs
            P100_demotion = list(set(P100_demote_list).intersection(sorted_pool))
            V100_demotion = list(set(V100_demote_list).intersection(sorted_pool))
            return P100_demotion, V100_demotion
       

def save_job(node, job): # save_job('c2176', '50')
    # first wait for the job to be qualified for checkpointing
    while True: # wait for ckpt_qual to be available
        global ckpt_qual_dict
        if ckpt_qual_dict['job'+job] == 1:
            ckpt_qual_dict['job'+job] = 0
            break
        time.sleep(5)
    
    global pid_dict
    pid = pid_dict['job'+job]
    send_signal(node, 'save ' + job + ' pid ' + pid) # 'save 50 pid 10000'

    global ovhd_start
    ovhd_start[job] = time.time()

    time.sleep(3) # in case epoch_waste is communicate too frequently

def kill_job(node, job): # kill_job('c2176', '50')
    send_signal(node, 'kill ' + job)

# resume job
def resume_job(node, gpu, job): # resume_job('c2176', '3', '50')
    cmd = 'resume ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

# start job
def start_job(node, gpu, job):
    cmd = 'start ' + job + ' gpu ' + gpu
    send_signal(node, cmd)   

# function that checks the tensorboard log of currently running jobs and logs jobs that have finished the first epoch
# in a global list. Once it's done, it will be in a queue to be promoted to V100 for 3 more epochs.
def check_step1_complete_V100(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    global step1_job_V100
    global V100_epoch_time
  
    for job in job_list:
        if job not in step1_job_V100 and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            tc = ''
            for item in dirs:
                item_node = item.split('/')[-1].split('.')[-1]
                if item_node in V100_node:
                    tc = item
            if tc != '':
                iterator = EventAccumulator(tc).Reload()
                tag = 'loss'
                try:
                    if len(iterator.Scalars(tag)) > 2: # this way we can collect one epoch time
                        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
                        V100_epoch_time[job] = wall_time[1] - wall_time[0]
                        step1_job_V100.append(job)
                        print('job' + job + ' has reached step1 complete on V100', file=run_log, flush=True)
                except Exception:
                    pass

def check_step1_complete_P100(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    global step1_job_P100
    global P100_epoch_time
  
    for job in job_list:
        if job not in step1_job_P100 and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            tc = ''
            for item in dirs:
                item_node = item.split('/')[-1].split('.')[-1]
                if item_node in P100_node:
                    tc = item
            if tc != '':
                iterator = EventAccumulator(tc).Reload()
                tag = 'loss'
                try:
                    if len(iterator.Scalars(tag)) > 2: # this way we can collect one epoch time
                        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
                        P100_epoch_time[job] = wall_time[1] - wall_time[0]
                        step1_job_P100.append(job)
                        print('job' + job + ' has reached step1 complete on P100', file=run_log, flush=True)
                except Exception:
                    pass


def check_step2_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    global step1_job_P100, step1_job_V100
    global step2_job
    global K80_epoch_time
    step1_job = list(set(step1_job_P100).union(step1_job_V100))

    for job in job_list:
        if job in step1_job and job not in step2_job and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            tc = ''
            for item in dirs:
                item_node = item.split('/')[-1].split('.')[-1]
                if item_node in K80_node:
                    tc = item
            if tc != '':
                iterator = EventAccumulator(tc).Reload()
                tag = 'loss'
                try:
                    if len(iterator.Scalars(tag)) > 2: # this way we can collect one epoch time
                        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
                        K80_epoch_time[job] = wall_time[1] - wall_time[0]
                        step2_job.append(job)
                        print('job' + job + ' has reached step2 complete on K80', file=run_log, flush=True)
                except Exception:
                    pass

# measure job
def measure_job(node, gpu, job):
    cmd = 'measure ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

############### first clear finish status of all jobs ####################

pid_dict = {}
for i in range(len(queue)):
    job_name = 'job' + str(i + 1)
    pid_dict[job_name] = 0

checkpoint_dict = {}
for i in range(len(queue)):
    job_name = 'job' + str(i + 1)
    checkpoint_dict[job_name] = 0

ckpt_qual_dict = {}
for i in range(len(queue)):
    job_name = 'job' + str(i + 1)
    ckpt_qual_dict[job_name] = 0

finish_dict = {}
for i in range(len(queue)):
    job_name = 'job' + str(i + 1)
    finish_dict[job_name] = 0

epoch_waste_dict = {}
for i in range(len(queue)):
    job_name = 'job' + str(i + 1)
    epoch_waste_dict[job_name] = 0

#################### background thread running TCP socket ########################

def thread_function():
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host_node, 10002)
    print('starting up on {} port {}'.format(*server_address), file=run_log, flush=True)
    sock.bind(server_address)
    sock.listen(5)  
    while True:
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    global K80_start_time, P100_start_time, V100_start_time, promote_start_time
                    global K80_job, P100_job, V100_job
                    global K80_time, P100_time, V100_time
                    global ovhd_a, ovhd_b, ovhd_c, ovhd_d, k80_1st, p100_1st, v100_1st, ovhd_start, overhead, ovhd_total
                    global b_start, c_start, d_start, completion
                    if 'ckpt_qual' in data_str:
                        global ckpt_qual_dict
                        job_name = data_str.split(' ')[0]
                        ckpt_qual_dict[job_name] = 1
                    elif 'finish' in data_str:
                        global finish_dict
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        finish_dict[job_name] = 1
                        JCT[job] = int(time.time() - job_start[job])
                        if job in list(K80_job.values()):
                            K80_time[job] += int(time.time() - K80_start_time[job])
                        elif job in list(P100_job.values()):
                            P100_time[job] += int(time.time() - P100_start_time[job])
                        elif job in list(V100_job.values()):
                            V100_time[job] += int(time.time() - V100_start_time[job])
                    elif 'pid' in data_str:
                        global pid_dict
                        job_name = data_str.split(' ')[0]
                        pid = data_str.split(' ')[2]
                        pid_dict[job_name] = pid
                    elif 'checkpoint' in data_str: # can only be received after save signal is sent
                        global checkpoint_dict
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')                        
                        checkpoint_dict[job_name] = 1
                        ovhd_a[job].append(int(time.time() - ovhd_start[job]))
                        b_start[job] = time.time()
                    elif 'waste' in data_str:
                        global epoch_waste_dict
                        job_name = data_str.split(' ')[0]
                        epoch_waste_time = data_str.split(' ')[2]
                        epoch_waste_dict[job_name] += int(epoch_waste_time)
                    elif 'b_end' in data_str:
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        ovhd_b[job].append(int(time.time() - b_start[job]))
                        c_start[job] = time.time()
                    elif 'c_end' in data_str:
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        ovhd_c[job].append(int(time.time() - c_start[job]))
                        d_start[job] = time.time()
                    elif 'd_end' in data_str:
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        ovhd_d[job].append(int(time.time() - d_start[job]))
                        ovhd_total[job].append(int(time.time() - ovhd_start[job]))
                        if ovhd_start[job] != 0:
                            overhead[job] += int(time.time() - ovhd_start[job])
                            ovhd_start[job] = 0 
                            if job in list(K80_job.values()):
                                K80_start_time[job] = time.time()
                            elif job in list(P100_job.values()):
                                P100_start_time[job] = time.time()
                            elif job in list(V100_job.values()):
                                V100_start_time[job] = time.time()
                                promote_start_time[job] = time.time()
                    elif '1st_epoch' in data_str: # 'job50 1st_epoch 35'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        epoch_time = int(data_str.split(' ')[2])
                        if job in list(K80_job.values()):
                            k80_1st[job].append(epoch_time)
                        elif job in list(P100_job.values()):
                            p100_1st[job].append(epoch_time)
                        elif job in list(V100_job.values()):
                            v100_1st[job].append(epoch_time)
                    elif 'completion' in data_str: # 'job50 completion 0.33'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        completion_portion = float(data_str.split(' ')[2])
                        completion[job] = completion_portion
                    #if 'ckpt_qual' in data_str or 'finish' in data_str or 'checkpoint' in data_str:
                    #    print('received ' + data_str)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()

x = threading.Thread(target=thread_function, daemon=True)
x.start()

###############################################################################

######################################################################

while True:
    
    # termination condition: 
    # all the jobs have finished

    ################### check for finished jobs on K80 and V100 ##############################

    for gpu, job in K80_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                K80_used -= 1            
                K80_job[gpu] = 'idle'
                print('K80 finished job: ' + job, file=run_log, flush=True)

    for gpu, job in P100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                P100_used -= 1            
                P100_job[gpu] = 'idle'
                print('P100 finished job: ' + job, file=run_log, flush=True)
                if job in P100_demote_list:
                    P100_demote_list.remove(job)

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job, file=run_log, flush=True)
                if job in V100_demote_list:
                    V100_demote_list.remove(job)

    ################ check step1 finished job of K80 jobs and step 2 of V100 #################

    check_step1_complete_V100(list(V100_job.values()))
    # make predictions for jobs finished step1 on V100, only once for each job's lifetime
    for gpu, job in V100_job.items():
        if job not in qualified_job and job != 'idle':
            if job in step1_job_V100:
                real_node, real_gpu = V100_LUT(gpu)
                kill_job(real_node, job)
                qualified_job.append(job)
                print('job' + job + ' has been qualified for demotion', file=run_log, flush=True)
                time.sleep(3) # wait for run.sh to finish
                x1, x3 = gpu_pwr.process_csv('job'+job, testcase)
                x2 = 3600 / V100_epoch_time[job]
                # preprocess the data
                x1 = (x1 - min(x1_v100)) / (max(x1_v100) - min(x1_v100))
                x2 = (x2 - min(x2_v100)) / (max(x2_v100) - min(x2_v100))
                x3 = (x3 - min(x3_v100)) / (max(x3_v100) - min(x3_v100))

                speedup_pred_KV = model_V100_KV.predict(np.array([x1, x2, x3]).reshape((1,-1)))[0] / 100
                speedup_dict_V100[job] = speedup_pred_KV
                predict_dict_V100[job] = speedup_pred_KV
                speedup_pred_KP = model_V100_KP.predict(np.array([x1, x2, x3]).reshape((1,-1)))[0] / 100
                speedup_dict_P100[job] = speedup_pred_KP
                predict_dict_P100[job] = speedup_pred_KP

    check_step1_complete_P100(list(P100_job.values()))
    # make predictions for jobs finished step1 on P100, only once for each job's lifetime
    for gpu, job in P100_job.items():
        if job not in qualified_job and job != 'idle':
            if job in step1_job_P100:
                real_node, real_gpu = P100_LUT(gpu)
                kill_job(real_node, job)
                qualified_job.append(job)
                print('job' + job + ' has been qualified for demotion', file=run_log, flush=True)
                time.sleep(3) # wait for run.sh to finish
                x1, x3 = gpu_pwr.process_csv('job'+job, testcase)
                x2 = 3600 / P100_epoch_time[job]
                # preprocess the data
                x1 = (x1 - min(x1_p100)) / (max(x1_p100) - min(x1_p100))
                x2 = (x2 - min(x2_p100)) / (max(x2_p100) - min(x2_p100))
                x3 = (x3 - min(x3_p100)) / (max(x3_p100) - min(x3_p100))

                speedup_pred_KP = model_P100_KP.predict(np.array([x1, x2, x3]).reshape((1,-1)))[0] / 100
                speedup_dict_P100[job] = speedup_pred_KP
                predict_dict_P100[job] = speedup_pred_KP
                speedup_pred_KV = model_P100_KV.predict(np.array([x1, x2, x3]).reshape((1,-1)))[0] / 100
                speedup_dict_V100[job] = speedup_pred_KV
                predict_dict_V100[job] = speedup_pred_KV

    check_step2_complete(list(K80_job.values()))   
    # correct speedup predictions
    for job in speedup_dict_V100:
        if speedup_dict_V100[job] != 0 and speedup_dict_V100[job] == predict_dict_P100[job]:
            if K80_epoch_time[job] != 0 and V100_epoch_time[job] != 0:
                speedup_dict_V100[job] = (K80_epoch_time[job] - V100_epoch_time[job]) / K80_epoch_time[job]
    for job in speedup_dict_P100:
        if speedup_dict_P100[job] != 0 and speedup_dict_P100[job] == predict_dict_P100[job]:
            if K80_epoch_time[job] != 0 and P100_epoch_time[job] != 0:
                speedup_dict_P100[job] = (K80_epoch_time[job] - P100_epoch_time[job]) / K80_epoch_time[job]

    ############### record number of newly arrived jobs ################

    new_arrival = 0
    index_cpy = index
    while True:
        time_passed = int(time.time() - queue_timer)
        if index_cpy >= len(queue):
            break
        elif time_passed >= queue_dict[queue[index_cpy]]:
            new_arrival += 1
            index_cpy += 1
        elif time_passed < queue_dict[queue[index_cpy]]:
            break

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    P100_free = P100_cap - P100_used
    K80_free = K80_cap - K80_used
    if new_arrival == 0:
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
        K80_promote_list = list(set(step2_job).intersection(list(K80_job.values())))
        if K80_free == K80_cap:
            P100_promote_list = list(set(qualified_job).intersection(list(P100_job.values())))
    else:
        K80_promote_list = list(set(step2_job).intersection(list(K80_job.values()))) # have to finish K80 profiling

    # look at demote list
    for gpu, job in V100_job.items():
        if job != 'idle' and job in step1_job_V100:
            if job not in V100_demote_list and job in step2_job and len(ovhd_total[job]) > 0:
                job_ovhd = np.mean(ovhd_total[job]) # 100
                if len(k80_1st[job]) > 0:
                    k80_1st_ovhd = np.mean(k80_1st[job]) - K80_epoch_time[job]
                else:
                    k80_1st_ovhd = 0
##                print('printing v100_1st ' + job + ' for debugging purpose: ' + str(v100_1st[job]))
                v100_1st_ovhd = np.mean(v100_1st[job]) - V100_epoch_time[job]

                demote_qualify_time_V100 = (2 * job_ovhd + k80_1st_ovhd + v100_1st_ovhd) / speedup_dict_V100[job]
                if int(time.time() - promote_start_time[job]) > max(demote_qualify_time_V100, max(v100_1st[job])):
                    V100_demote_list.append(job)
                    print('job' + job + 'qualified for demote for passing demote qualify time', file=run_log, flush=True)
                    ##str(int(demote_qualify_time)))
            elif job not in V100_demote_list and job not in step2_job and job in qualified_job:
                V100_demote_list.append(job)
                print('job' + job + 'qualified for demote for profiling', file=run_log, flush=True)

    for gpu, job in P100_job.items():
        if job != 'idle' and job in step1_job_P100:
            if job not in P100_demote_list and job in step2_job and len(ovhd_total[job]) > 0:
                job_ovhd = np.mean(ovhd_total[job]) # 100
                if len(k80_1st[job]) > 0:
                    k80_1st_ovhd = np.mean(k80_1st[job]) - K80_epoch_time[job]
                else:
                    k80_1st_ovhd = 0

                p100_1st_ovhd = np.mean(p100_1st[job]) - P100_epoch_time[job]

                demote_qualify_time_P100 = (2 * job_ovhd + k80_1st_ovhd + p100_1st_ovhd) / speedup_dict_P100[job]
                if int(time.time() - promote_start_time[job]) > max(demote_qualify_time_P100, max(p100_1st[job])):
                    P100_demote_list.append(job)
                    print('job' + job + 'qualified for demote for passing demote qualify time', file=run_log, flush=True)
                    ##str(int(demote_qualify_time)))
            elif job not in P100_demote_list and job not in step2_job and job in qualified_job:
                P100_demote_list.append(job)
                print('job' + job + 'qualified for demote for profiling', file=run_log, flush=True)

    if len(K80_promote_list) > 0 or len(P100_demote_list) > 0 or len(V100_demote_list) > 0:
        if new_arrival == 0 and K80_free < K80_cap: # all jobs received, jobs still running on K80
            promote_list_cpy = K80_promote_list[:]
            V100_promoted, V100_demoted = max_speedup_promotion_V100(V100_free, promote_list_cpy, V100_demote_list)
            if len(V100_demoted) > len(V100_promoted):
                print('should never happen for K80', file=run_log, flush=True)
                pdb.set_trace()
            promote_list_cpy = list(set(promote_list_cpy).difference(V100_promoted))
            P100_promoted, P100_demoted = max_speedup_promotion_P100(P100_free, promote_list_cpy, P100_demote_list)
            if len(P100_demoted) > len(P100_promoted):
                print('should never happen for P100', file=run_log, flush=True)
                pdb.set_trace()
        elif new_arrival == 0 and K80_free == K80_cap: # all jobs received, jobs only running on P100 and V100
            # when there are free V100s, promote P100 job to V100
            V100_promoted, V100_demoted = max_speedup_promotion_P2V(V100_free, P100_promote_list)
            P100_promoted = []
            P100_demoted = []
           
        else:
            # first demote P100 jobs, then demote V100 jobs
            P100_demote_list_cpy = P100_demote_list[:]
            V100_demote_list_cpy = V100_demote_list[:]
            promote_list_cpy = K80_promote_list[:]
            P100_promoted, P100_demoted = min_speedup_demotion_P100(promote_list_cpy, P100_demote_list_cpy)

            promote_list_cpy = list(set(promote_list_cpy).difference(P100_promoted))
            V100_promoted, V100_demoted = min_speedup_demotion_V100(promote_list_cpy, V100_demote_list_cpy)

            P100_demote_list_cpy = list(set(P100_demote_list_cpy).difference(P100_demoted))
            V100_demote_list_cpy = list(set(V100_demote_list_cpy).difference(V100_demoted))
            # also consider demote newly promoted job to free K80
            P100_demote_list_cpy = list(set(P100_demote_list_cpy).union(P100_promoted))
            V100_demote_list_cpy = list(set(V100_demote_list_cpy).union(V100_promoted))

            have_to_demote = new_arrival - V100_free - P100_free
            P100_demoted_free, V100_demoted_free = min_speedup_demotion_free(K80_free, P100_demote_list_cpy,
            V100_demote_list_cpy, have_to_demote)
            # remove jobs that got demoted back immediately after promotion
            V100_promoted = list(set(V100_promoted).difference(V100_demoted_free))
            P100_promoted = list(set(P100_promoted).difference(P100_demoted_free))
            # 1st phase demoted and 2nd phase demote to free
            V100_demoted = list(set(V100_demoted).union(V100_demoted_free))
            P100_demoted = list(set(P100_demoted).union(P100_demoted_free))
            # make sure demoted jobs are not already in K80 promote list
            V100_demoted = list(set(V100_demoted).difference(K80_promote_list))
            P100_demoted = list(set(P100_demoted).difference(K80_promote_list))

        total_demoted = list(set(V100_demoted).union(P100_demoted))
        total_promoted = list(set(V100_promoted).union(P100_promoted))

        if len(V100_promoted) > 0:
            if new_arrival == 0:
                print('no new job arrivals', file=run_log, flush=True)
                if K80_free == K80_cap:
                    print('jobs promoted from P100 to V100', file=run_log, flush=True)
            print('V100 promoted jobs: ', V100_promoted, file=run_log, flush=True)
        if len(P100_promoted) > 0:
            if new_arrival == 0:
                print('no new job arrivals', file=run_log, flush=True)
            print('P100 promoted jobs: ', P100_promoted, file=run_log, flush=True)

        if len(V100_demoted) > 0:
            print('V100 demoted jobs: ', V100_demoted, file=run_log, flush=True)
        if len(P100_demoted) > 0:
            print('P100 demoted jobs: ', P100_demoted, file=run_log, flush=True)
        # stop all promoted jobs on K80
        checkpoint_finish_check = []
        for gpu, job in K80_job.items():
            if job in total_promoted:
                real_node, real_gpu = K80_LUT(gpu)
                save_job(real_node, job)
                if finish_dict['job'+job] != 1:
                    K80_time[job] += int(time.time() - K80_start_time[job])
                checkpoint_finish_check.append(job)
                K80_job[gpu] = 'idle'
                K80_used -= 1
        # stop all demoted jobs on P100
        for gpu, job in P100_job.items():
            if job in total_demoted:
                real_node, real_gpu = P100_LUT(gpu)
                save_job(real_node, job)
                if finish_dict['job'+job] != 1:
                    P100_time[job] += int(time.time() - P100_start_time[job])
                checkpoint_finish_check.append(job)
                P100_job[gpu] = 'idle'
                P100_used -= 1
                P100_demote_list.remove(job)
        if new_arrival == 0 and K80_free == K80_cap:
            # stop all promoted jobs on P100
            checkpoint_finish_check = []
            for gpu, job in P100_job.items():
                if job in total_promoted:
                    real_node, real_gpu = P100_LUT(gpu)
                    save_job(real_node, job)
                    if finish_dict['job'+job] != 1:
                        P100_time[job] += int(time.time() - P100_start_time[job])
                    checkpoint_finish_check.append(job)
                    P100_job[gpu] = 'idle'
                    P100_used -= 1
           
        # stop all demoted jobs on V100
        for gpu, job in V100_job.items():
            if job in total_demoted:
                real_node, real_gpu = V100_LUT(gpu)
                save_job(real_node, job)
                if finish_dict['job'+job] != 1:
                    V100_time[job] += int(time.time() - V100_start_time[job])
                checkpoint_finish_check.append(job)
                V100_job[gpu] = 'idle'
                V100_used -= 1
                V100_demote_list.remove(job)

        # wait for all GPUs to be available
        if len(checkpoint_finish_check) > 0:
            while True:
                time.sleep(5)
                for job in checkpoint_finish_check[:]:
                    if checkpoint_dict['job'+job] == 1: # checkpoint has finished, gpu is free
                        print(job + ' checkpointed successfully', file=run_log, flush=True)
                        checkpoint_dict['job'+job] = 0 # reset it
                        checkpoint_finish_check.remove(job)
                    # also check if job already finished before sending checkpoint signal
                    elif finish_dict['job'+job] == 1:
                        print(job + ' finished before receiving checkpoint signal', file=run_log, flush=True)
                        checkpoint_finish_check.remove(job)
                if len(checkpoint_finish_check) == 0:
                    break
        # give it some time to cleanup old checkpointed jobs
        time.sleep(3)
        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in V100_promoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in V100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        V100_job[gpu] = job_new
                        real_node, real_gpu = V100_LUT(gpu)                           
                        resume_job(real_node, real_gpu, job_new)
                        num_mig[job_new] += 1
                        total_promoted.remove(job_new)
                        V100_used += 1
                        break
            else: # job has already finished before checkpointing
                total_promoted.remove(job_new)

        # resume promoted jobs on P100, make sure the gpu is idle
        for job_new in P100_promoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in P100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        P100_job[gpu] = job_new
                        real_node, real_gpu = P100_LUT(gpu)                           
                        resume_job(real_node, real_gpu, job_new)
                        num_mig[job_new] += 1
                        total_promoted.remove(job_new)
                        P100_used += 1
                        break
            else: # job has already finished before checkpointing
                total_promoted.remove(job_new)

        # resume demoted jobs on K80, make sure the gpu is idle
        for job_new in total_demoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in K80_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        real_node, real_gpu = K80_LUT(gpu)
                        resume_job(real_node, real_gpu, job_new)
                        num_mig[job_new] += 1
                        K80_job[gpu] = job_new
                        total_demoted.remove(job_new)
                        K80_used += 1
                        break
            else: # job has already finished before checkpointing
                print('job'+job_new+' has finished before checkpointing', file=run_log, flush=True)
                total_demoted.remove(job_new)

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(total_promoted) > 0 or len(total_demoted) > 0:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')

    ################ submit new jobs to vacant GPUs ############################

    if not all_jobs_started:
        if V100_used < V100_cap:
            V100_free = V100_cap - V100_used
            for i in range(V100_free):
                time_passed = int(time.time() - queue_timer)
                if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                    job_new = str(queue[index])
                    for gpu, job in V100_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = V100_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            birthplace[job_new] = real_node
                            measure_job(real_node, real_gpu, job_new)
                            V100_job[gpu] = job_new
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])                    
                            V100_start_time[job_new] = time.time()
                            index += 1
                            V100_used += 1
                            time.sleep(5) # don't communicate too often
                            break
                elif index >= len(queue):
                    all_jobs_started = True
    if not all_jobs_started:
        if P100_used < P100_cap:
            P100_free = P100_cap - P100_used
            for i in range(P100_free):
                time_passed = int(time.time() - queue_timer)
                if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                    job_new = str(queue[index])
                    for gpu, job in P100_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = P100_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            birthplace[job_new] = real_node
                            measure_job(real_node, real_gpu, job_new)
                            P100_job[gpu] = job_new
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])                    
                            P100_start_time[job_new] = time.time()
                            index += 1
                            P100_used += 1
                            time.sleep(5) # don't communicate too often
                            break
                elif index >= len(queue):
                    all_jobs_started = True

    ############## monitor GPU usage ############

    usage = K80_used + P100_used + V100_used
    time_stamp = int(time.time() - queue_timer)
    gpu_usage_time.append(time_stamp)
    gpu_usage.append(usage)
    total_completion = np.sum(list(completion.values()))
    gpu_usage_completion.append(total_completion)

    ############### wait for next iteration

    time.sleep(INTERVAL)

    ################ check if termination condition is met ################

    K80_idle_num = sum(value == 'idle' for value in K80_job.values())
    P100_idle_num = sum(value == 'idle' for value in P100_job.values())
    V100_idle_num = sum(value == 'idle' for value in V100_job.values())
    if K80_idle_num == K80_cap and P100_idle_num == P100_cap and V100_idle_num == V100_cap and index == len(queue):
        print('all jobs are finished!', file=run_log, flush=True)
        break


# get average JCT
average_JCT = np.average(list(JCT.values()))
JCT['average'] = average_JCT

average_overhead = np.average(list(overhead.values()))
overhead['average'] = average_overhead

average_queue_delay = np.average(list(queue_delay.values()))
queue_delay['average'] = average_queue_delay

# after everything is finished

print('finished all runs', file=run_log, flush=True)
JCT_name = testcase + '_JCT.json'
overhead_name = testcase + '_overhead.json'
num_mig_name = testcase + '_num_mig.json'
epoch_waste_name = testcase + '_epoch_waste.json'
ckpt_qual_name = 'ckpt_qual.json'
finish_name = 'finish.json'
K80_time_name = testcase + '_K80_time.json'
P100_time_name = testcase + '_P100_time.json'
V100_time_name = testcase + '_V100_time.json'
gpu_usage_name = testcase + '_gpu_usage.csv'
ovhd_a_name = testcase + '_ovhd_a.json'
ovhd_b_name = testcase + '_ovhd_b.json'
ovhd_c_name = testcase + '_ovhd_c.json'
ovhd_d_name = testcase + '_ovhd_d.json'
ovhd_total_name = testcase + '_ovhd_total.json'
k80_1st_name = testcase + '_k80_1st.json'
p100_1st_name = testcase + '_p100_1st.json'
v100_1st_name = testcase + '_v100_1st.json'
speedup_name_V100 = 'speedup_V100.json'
speedup_name_P100 = 'speedup_P100.json'
predict_name_V100 = 'predict_V100.json'
predict_name_P100 = 'predict_P100.json'
V100_demote_list_name = 'V100_demote_list.json'
P100_demote_list_name = 'P100_demote_list.json'

completion_name = 'completion.json'
queue_delay_name = testcase + '_queue_delay.json'
birthplace_name = testcase + '_birthplace.json'

with open(JCT_name, 'w') as fp1:
    json.dump(JCT, fp1, sort_keys=True, indent=4)
with open(overhead_name, 'w') as fp3:
    json.dump(overhead, fp3, sort_keys=True, indent=4)
with open(num_mig_name, 'w') as fp3:
    json.dump(num_mig, fp3, sort_keys=True, indent=4)
with open(epoch_waste_name, 'w') as fp3:
    json.dump(epoch_waste_dict, fp3, sort_keys=True, indent=4)
with open(ckpt_qual_name, 'w') as fp1:
    json.dump(ckpt_qual_dict, fp1, sort_keys=True, indent=4)
with open(finish_name, 'w') as fp1:
    json.dump(finish_dict, fp1, sort_keys=True, indent=4)
with open(K80_time_name, 'w') as fp3:
    json.dump(K80_time, fp3, sort_keys=True, indent=4)
with open(P100_time_name, 'w') as fp3:
    json.dump(P100_time, fp3, sort_keys=True, indent=4)
with open(V100_time_name, 'w') as fp3:
    json.dump(V100_time, fp3, sort_keys=True, indent=4)
with open(ovhd_a_name, 'w') as fp3:
    json.dump(ovhd_a, fp3, sort_keys=True, indent=4)
with open(ovhd_b_name, 'w') as fp3:
    json.dump(ovhd_b, fp3, sort_keys=True, indent=4)
with open(ovhd_c_name, 'w') as fp3:
    json.dump(ovhd_c, fp3, sort_keys=True, indent=4)
with open(ovhd_d_name, 'w') as fp3:
    json.dump(ovhd_d, fp3, sort_keys=True, indent=4)
with open(ovhd_total_name, 'w') as fp3:
    json.dump(ovhd_total, fp3, sort_keys=True, indent=4)
with open(k80_1st_name, 'w') as fp3:
    json.dump(k80_1st, fp3, sort_keys=True, indent=4)
with open(p100_1st_name, 'w') as fp3:
    json.dump(p100_1st, fp3, sort_keys=True, indent=4)
with open(v100_1st_name, 'w') as fp3:
    json.dump(v100_1st, fp3, sort_keys=True, indent=4)
with open(speedup_name_V100, 'w') as fp1:
   json.dump(speedup_dict_V100, fp1, sort_keys=True, indent=4)
with open(speedup_name_P100, 'w') as fp1:
   json.dump(speedup_dict_P100, fp1, sort_keys=True, indent=4)
with open(predict_name_V100, 'w') as fp1:
   json.dump(predict_dict_V100, fp1, sort_keys=True, indent=4)
with open(predict_name_P100, 'w') as fp1:
   json.dump(predict_dict_P100, fp1, sort_keys=True, indent=4)
with open(V100_demote_list_name, 'w') as fp1:
   json.dump(V100_demote_list, fp1, sort_keys=True, indent=4)
with open(P100_demote_list_name, 'w') as fp1:
   json.dump(P100_demote_list, fp1, sort_keys=True, indent=4)

with open(completion_name, 'w') as fp1:
   json.dump(completion, fp1, sort_keys=True, indent=4)
with open(queue_delay_name, 'w') as fp1:
   json.dump(queue_delay, fp1, sort_keys=True, indent=4)
with open(birthplace_name, 'w') as fp1:
   json.dump(birthplace, fp1, sort_keys=True, indent=4)

gpu_usage_time = np.asarray(gpu_usage_time)
gpu_usage = np.asarray(gpu_usage)
gpu_usage_completion = np.asarray(gpu_usage_completion)
rows = zip(gpu_usage_time, gpu_usage, gpu_usage_completion)
with open(gpu_usage_name, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

