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
import dash_opt

parser = argparse.ArgumentParser(description='TCP client')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()

with open('../job_trace/job_queue_50.json', 'r') as fp:
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

multigpu_list = ['1', '2', '3'] #TODO

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

num_mig = {} # initialize migration time to 0
for item in queue:
    num_mig[str(item)] = 0
queue_start = {} # initialize this to 0 as well
for item in queue:
    queue_start[str(item)] = 0
queue_time = {} # initialize this to 0 as well
for item in queue:
    queue_time[str(item)] = 0
#TODO: monitor batch time here
V100_batch_time = {}
for item in queue:
    V100_batch_time[str(item)] = 0
K80_batch_time = {}
for item in queue:
    K80_batch_time[str(item)] = 0

#TODO: monitor 1st epoch overhead after migration here
V100_1st_ovhd = {}
for item in queue:
    V100_1st_ovhd[str(item)] = 0
K80_1st_ovhd = {}
for item in queue:
    K80_1st_ovhd[str(item)] = 0

K80_start_time = {}
for item in queue:
    K80_start_time[str(item)] = 0
V100_start_time = {}
for item in queue:
    V100_start_time[str(item)] = 0

#TODO: may not be needed
promote_start_time = {}
for item in queue:
    promote_start_time[str(item)] = 0
demote_list = []

#TODO: most important new change
# this remaining time should be normalized to remaining time when running on K80
job_remaining_batch = {}
for item in queue:
    job_remaining_batch[str(item)] = 0

K80_time = {}
for item in queue:
    K80_time[str(item)] = 0
V100_time = {}
for item in queue:
    V100_time[str(item)] = 0
gpu_usage_time = [] # don't initialize this
gpu_usage = []
gpu_usage_completion = []

# TODO: here speedup should just be K80_time / V100_time
speedup_dict = {}
for item in queue:
    speedup_dict[str(item)] = 0 

birthplace = {}
for item in queue:
    birthplace[str(item)] = 'none' 

index = 0

K80_cap = 8
V100_cap = 4
K80_used = 0
V100_used = 0
K80_per_node = 8
V100_per_node = 4

K80_job = {}
for i in range(K80_cap):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(V100_cap):
    V100_job[str(i)] = 'idle'
step1_job = []
step2_job = []
pc_job = []

K80_node = ['c2177']
V100_node = ['d1018']
host_node = 'c0154'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s

# function to detect if there are two free or reserved GPUs in a node
# returns an empty list if there is none, otherwise returns list with gpu id in V100/K80_jobs
def detect_2_gpus(gpu_dict, gpu_per_node, status='idle'):
    job_list = list(gpu_dict.values())
    num_nodes = int(len(job_list) / gpu_per_node)
    for i in range(num_nodes):
        start = i * gpu_per_node
        end = i + gpu_per_node
        sliced_list = job_list[start:end]
        occurence = sliced_list.count(status)
        if occurence >= 2:
            # only take the first two elements
            indexs = [j for j, e in enumerate(sliced_list) if e == status]
            return [str(j + start) for j in indexs]
    return []

def K80_LUT(gpu):
    quotient = int(gpu) // 8
    remainder = int(gpu) % 8
    real_node = K80_node[quotient]
    real_gpu = str(remainder)
    return real_node, real_gpu
def V100_LUT(gpu):
    quotient = int(gpu) // 4
    remainder = int(gpu) % 4
    real_node = V100_node[quotient]
    real_gpu = str(remainder)
    return real_node, real_gpu

def send_signal(node, cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 10000 
    # Connect the socket to the port where the server is listening
    server_address = (node, int(port))

    print('connecting to {} port {}'.format(*server_address))
    sock.connect(server_address)

    try:
        # Send data
        message = cmd.encode('utf-8') #b'save 35'  #b'start 35 gpu 6'#b'save 35'
 
        print('sending {!r}'.format(message))
        sock.sendall(message)
        while True:
            data = sock.recv(32)
            if 'success' in data.decode('utf-8'):
#                print('received {!r}'.format(data))
                break
            else:
                print('waiting for success signal')
                time.sleep(1)
    finally:
        #print('closing socket')
        sock.close()

def get_avail_id(gpu_dict):
    # input is K80_job or V100_job (dict)
    key_list = list(gpu_dict.keys())
    value_list = list(gpu_dict.values())
    indexs = [j for j, e in enumerate(value_list) if e == 'idle']
    return [key_list[j] for j in indexs]

# 2-gpu jobs in new_pool have duplicated items
# returns mapping of jobs in "new_pool" to GPUs
def GPU_placement(GPU_avail, new_pool, gpu_type='K80', raise_error=True):
    mapping = {}
    skip = False
    res_group = [] # group reserved GPU together
    for i in range(len(GPU_avail)):
        if skip:
            skip = False
            continue
        else:
            # two gpus from the same node
            if gpu_type == 'K80':
                GPU_per_node = K80_per_node
            elif gpu_type == 'V100':
                GPU_per_node = V100_per_node
            if i!=len(GPU_avail)-1 and int(GPU_avail[i])//GPU_per_node==int(GPU_avail[i+1])//GPU_per_node:
                skip = True
                res_group.append([GPU_avail[i], GPU_avail[i+1]])
            else:
                res_group.append([GPU_avail[i]])
    group_1gpu = [i for i in res_group if len(i) == 1] # 1gpu id
    group_2gpu = [i for i in res_group if len(i) == 2] # 2gpu id
    pool_1gpu = [i for i in new_pool if i not in multigpu_list] # 1gpu job
    pool_2gpu = [i for i in new_pool if i in multigpu_list] # 2gpu job
    if len(GPU_avail) < len(new_pool) or 2*len(group_2gpu) < len(pool_2gpu):
        if raise_error:
            if gpu_type == 'K80':
                raise ValueError('Bug with K80 placement for new jobs, more jobs than free gpus')
            elif gpu_type == 'V100':
                raise ValueError('Bug with V100 placement for new jobs, more jobs than free gpus')
        else:
            return mapping
    # if there is no 2-gpu job
    if set(new_pool).isdisjoint(multigpu_list):
        for i in range(len(new_pool)):
            mapping[new_pool[i]] = GPU_avail[i]
    else:
        # first, fill in all 1gpu slots with 1-gpu jobs as much as possible
        for i in group_1gpu[:]:
            if len(pool_1gpu) > 0:
                mapping[pool_1gpu[0]] = i[0]
                pool_1gpu.pop(0)
        for i in group_2gpu[:]:
            if len(pool_2gpu) > 1:
                mapping[pool_2gpu[0]] = ','.join(i)
                pool_2gpu = [i for i in pool_2gpu if i != pool_2gpu[0]]
            elif len(pool_1gpu) > 0:
                mapping[pool_1gpu[0]] = i[0]
                if len(pool_1gpu) > 1:
                    mapping[pool_1gpu[1]] = i[1]
                    pool_1gpu.pop(1)
                pool_1gpu.pop(0)
    return mapping

#aa = K80_placement(['0','1','2','3','4'], ['3','3','1','1','50'])

# input: a list of jobs
# output: a dict of jobs with their remaining time on K80 and V100
# the remaining time on the other GPU type need to include migration overhead 
# 1. ovhd_total: the mean is average migration overhead once
# 2. 1st_ovhd: extra time spent on 1st epoch after migration
# the returned dict looks like this {'50': [300, 150], '78': [1000, 300]}
# if a job can't be migrated yet (not in step1_job list) it shouldn't be in the input list
# elif a job can be migrated but have not been migrated or have been migration but does not have speedup yet
# , it should have the other gpu type remaining time as migration overhead

def get_remaining_time(job_list):
    result_dict = {}
    for job in job_list:
        if job not in step1_job:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')
        elif job in step1_job and job not in step2_job:
            K80_remain = job_remaining_batch[job] * K80_batch_time[job]
            V100_remain = job_remaining_batch[job] * V100_batch_time[job]
            # this is not accurate, but just to force job to run on the other GPU type not profiled
            if birthplace[job] in K80_node:
                result_dict[job] = [2 * K80_remain, 0]
            elif birthplace[job] in V100_node:
                result_dict[job] = [0, 2 * V100_remain]
        else: # job has its K80_batch_time and V100_batch_time profiled
            K80_remain = job_remaining_batch[job] * K80_batch_time[job]
            V100_remain = job_remaining_batch[job] * V100_batch_time[job]
            K80_mig_ovhd = np.mean(ovhd_total[job]) + K80_1st_ovhd[job]
            V100_mig_ovhd = np.mean(ovhd_total[job]) + V100_1st_ovhd[job]
            if job in list(K80_job.values()):
                result_dict[job] = [K80_remain, V100_remain + V100_mig_ovhd]
            elif job in list(V100_job.values()):
                result_dict[job] = [K80_remain + K80_mig_ovhd, V100_remain]
    return result_dict

#d, e, f = random_promotion(['0','1','4','8'], ['3','3','1','1'], [])

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

# resume job
def resume_job(node, gpu, job): # resume_job('c2176', '3', '50')
    cmd = 'resume ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

# start job
def start_job(node, gpu, job):
    cmd = 'start ' + job + ' gpu ' + gpu
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
    print('starting up on {} port {}'.format(*server_address))
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
                    global K80_start_time
                    global V100_start_time
                    global K80_job
                    global V100_job
                    global K80_time
                    global V100_time
                    global ovhd_a, ovhd_b, ovhd_c, ovhd_d, ovhd_start, overhead, ovhd_total
                    global b_start, c_start, d_start, completion
                    global step1_job, step2_job
                    global V100_batch_time, K80_batch_time, job_remaining_batch, speedup_dict
                    global K80_1st_ovhd, V100_1st_ovhd
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
                            elif job in list(V100_job.values()):
                                V100_start_time[job] = time.time()
                    elif 'completion' in data_str: # 'job50 completion 0.33'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        completion_portion = float(data_str.split(' ')[2])
                        completion[job] = completion_portion
                    elif 'batch_time' in data_str: # 'job50 batch_time 0.042'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        batch_time = float(data_str.split(' ')[2])
                        # also step1_job and step2_job
                        # if job birthplace is K80, K80_batch_time is collected, then step1 complete
                        if job in list(K80_job.values()) and K80_batch_time[job] == 0:
                            K80_batch_time[job] = batch_time
                            if birthplace[job] in K80_node:
                                step1_job.append(job)
                            elif birthplace[job] in V100_node:
                                step2_job.append(job)
                                speedup_dict[job] = round(K80_batch_time[job] / V100_batch_time[job], 3)
                        elif job in list(V100_job.values()) and V100_batch_time[job] == 0:
                            V100_batch_time[job] = batch_time
                            if birthplace[job] in V100_node:
                                step1_job.append(job)
                            elif birthplace[job] in K80_node:
                                step2_job.append(job)
                                speedup_dict[job] = round(K80_batch_time[job] / V100_batch_time[job], 3)
                    elif 'remain_batch' in data_str: # 'job50 remain_batch 156300'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        remaining_batch = int(data_str.split(' ')[2])
                        job_remaining_batch[job] = remaining_batch 
                    elif '1st_ovhd' in data_str: # 'job50 1st_ovhd 4.99'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        ovhd_time = float(data_str.split(' ')[2])
                        if job in list(K80_job.values()) and K80_1st_ovhd[job] == 0:
                            K80_1st_ovhd[job] = ovhd_time
                        elif job in list(V100_job.values()) and V100_1st_ovhd[job] == 0:
                            V100_1st_ovhd[job] = ovhd_time
                       
#                    if 'ckpt_qual' in data_str or 'finish' in data_str or 'checkpoint' in data_str:
#                        print('received ' + data_str)
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
                print('K80 finished job: ' + job)

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job)

    ################ submit new jobs to vacant K80 GPUs ############################

    # first fill in vacant V100s
    if V100_used < V100_cap:
        V100_free = V100_cap - V100_used
        for i in range(V100_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                if job_new in multigpu_list:
                    # find 2 gpus in the same node to schedule it
                    idle_gpus = detect_2_gpus(V100_job, V100_per_node)
                    if len(idle_gpus) > 0:
                        node_string = ''
                        for gpu in idle_gpus:
                            real_node, real_gpu = V100_LUT(gpu)
                            if gpu == idle_gpus[1]:
                                gpu_str += real_gpu
                                node_string = real_node
                                job_start[job_new] = time.time()
                                queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                                V100_start_time[job_new] = time.time()
                                index += 1
                            else:
                                gpu_str = real_gpu + ','
                            V100_job[gpu] = job_new
                            V100_used += 1
                        start_job(node_string, gpu_str, job_new)
                        birthplace[job_new] = node_string
                        time.sleep(5) # don't communicate too often
                else:
                    for gpu, job in V100_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = V100_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            birthplace[job_new] = real_node
                            V100_job[gpu] = job_new
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                            V100_start_time[job_new] = time.time()
                            index += 1
                            V100_used += 1
                            time.sleep(5) # don't communicate too often
                            break
    # first fill in vacant K80s
    if K80_used < K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                if job_new in multigpu_list:
                    # find 2 gpus in the same node to schedule it
                    idle_gpus = detect_2_gpus(K80_job, K80_per_node)
                    if len(idle_gpus) > 0:
                        node_string = ''
                        for gpu in idle_gpus:
                            real_node, real_gpu = K80_LUT(gpu)
                            if gpu == idle_gpus[1]:
                                gpu_str += real_gpu
                                node_string = real_node
                                job_start[job_new] = time.time()
                                queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                                K80_start_time[job_new] = time.time()
                                index += 1
                            else:
                                gpu_str = real_gpu + ','
                            K80_job[gpu] = job_new
                            K80_used += 1
                        start_job(node_string, gpu_str, job_new)
                        birthplace[job_new] = node_string
                        time.sleep(5) # don't communicate too often
                else:
                    for gpu, job in K80_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = K80_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            birthplace[job_new] = real_node
                            K80_job[gpu] = job_new
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                            K80_start_time[job_new] = time.time()
                            index += 1
                            K80_used += 1
                            time.sleep(5) # don't communicate too often
                            break

    ################## make promotion decisions ################
    # figure out which job enters the pool and which GPUs enter the pool
    # job must be in step1_job, and if it's on V100, it must have passed demote_qualify_time (TODO: demote_list)
    # the selected job's current GPU also enters GPU pool. And if GPU is idle, it gets added into the pool as well
    job_pool = []
    K80_pool = []
    V100_pool = []
    for gpu, job in K80_job.items():
        if job in step1_job:
            if job not in job_pool: # for 2-gpu jobs, add the job once, but add both gpus
                job_pool.append(job)
            K80_pool.append(gpu)
        elif job == 'idle':
            K80_pool.append(gpu)
    for gpu, job in V100_job.items():
        if job in step1_job:
            if job not in job_pool: # for 2-gpu jobs, add the job once, but add both gpus
                job_pool.append(job)
            V100_pool.append(gpu)
        elif job == 'idle':
            V100_pool.append(gpu)
    
    # prepare inputs and perform optimization
    num_GPUs = [len(K80_pool), len(V100_pool)]
    job_num_GPUs = {}
    for job in job_pool:
        if job in multigpu_list:
            job_num_GPUs[job] = 2
        else:
            job_num_GPUs[job] = 1
    job_remaining_time = get_remaining_time(job_pool)

    # perform 1st optimization
    opt_decision = dash_opt.optimize_promotion(num_GPUs, job_num_GPUs, job_remaining_time)

    # check if placement of promo/demo 2-gpu jobs are viable
    # if not viable: remove jobs that benefit least from promo/hurt least from demo
    promoted = [] # jobs to be placed in V100. 2-gpu jobs are duplicated
    demoted = [] # jobs to be placed in K80
    for job, placement in opt_decision:
        if placement == 1 and job in list(K80_job.values()):
            promoted.append(job)
            # duplicate the job if it's 2-gpu job
            if job in multigpu_list:
                promoted.append(job)
        elif placement == 0 and job in list(V100_job.values()):
            demoted.append(job)
            # duplicate the job if it's 2-gpu job
            if job in multigpu_list:
                demoted.append(job)

    if len(promoted) > 0:
        print('original promotion (2-gpu dup)', promoted)
    if len(demoted) > 0:
        print('original demotion (2-gpu dup)', demoted)

    # generate K80/V100 GPU list that are either idle or have job in promoted/demoted
    # to be used by placement function
    K80_avail = []
    V100_avail = []
    for gpu, job in K80_job.items():
        if job == 'idle':
            K80_avail.append(gpu)
        elif job in promoted:
            K80_avail.append(gpu)
    for gpu, job in V100_job.items():
        if job == 'idle':
            V100_avail.append(gpu)
        elif job in promoted:
            V100_avail.append(gpu)

    V100_place = GPU_placement(V100_avail, promoted, gpu_type='V100', raise_error=False)
    K80_place = GPU_placement(K80_avail, demoted, gpu_type='K80', raise_error=False)

    if V100_place == {} or K80_place == {}:
        print('not enough local 2-gpu slots to perform this optimization')
        # there are promoted/demoted 2-gpu job that don't fit in target GPU with locality constraint
        if V100_place == {}: # need to remove 2-gpu job from promoted
            #TODO: or come up with a new constraint

    # make promotion decisions
    if len(promote_list) > 0 or len(new_pool) > 0:
        # started and promoted do not have duplicated elements
        started, promoted, mapping = random_promotion(V100_avail, new_pool, promote_list)
        # stop all promoted jobs on K80
        checkpoint_finish_check = []
        for job in promoted[:]:
            # need to find its current gpu on K80
            current_gpu = ''
            for gpu, job_K in K80_job.items():
                if job_K == job:
                    current_gpu = gpu
                    break
            real_node, real_gpu = K80_LUT(current_gpu)
            save_job(real_node, job)
            if finish_dict['job'+job] != 1:
                K80_time[job] += int(time.time() - K80_start_time[job])
            checkpoint_finish_check.append(job)
            K80_job[current_gpu] = 'idle'
            K80_used -= 1

        # wait for all GPUs to be available
        if len(checkpoint_finish_check) > 0:
            while True:
                time.sleep(5)
                for job in checkpoint_finish_check[:]:
                    if checkpoint_dict['job'+job] == 1: # checkpoint has finished, gpu is free
                        print(job + ' checkpointed successfully')
                        checkpoint_dict['job'+job] = 0 # reset it
                        checkpoint_finish_check.remove(job)
                    # also check if job already finished before sending checkpoint signal
                    elif finish_dict['job'+job] == 1:
                        print(job + ' finished before receiving checkpoint signal')
                        checkpoint_finish_check.remove(job)
                if len(checkpoint_finish_check) == 0:
                    break
        # give it some time to cleanup old checkpointed jobs
        time.sleep(3)

        # 1. deal with all V100 jobs (started, promoted). The job-gpu mapping is already known 
        for job in started[:]: # new jobs
            gpu = mapping[job]
            if job not in multigpu_list:
                real_node, real_gpu = V100_LUT(gpu)
                start_job(real_node, real_gpu, job)
                V100_job[gpu] = job
                job_start[job] = time.time()
                queue_delay[job] = int(time.time() - queue_timer - queue_dict[int(job)])
                V100_start_time[job] = time.time()
                new_pool.remove(job)
            else:
                gpu_split = gpu.split(',')
                node_string = ''
                for g in gpu_split:
                    real_node, real_gpu = V100_LUT(g)
                    if g == gpu_split[1]:
                        gpu_str += real_gpu
                        node_string = real_node
                        job_start[job] = time.time()
                        queue_delay[job] = int(time.time() - queue_timer - queue_dict[int(job)])
                        V100_start_time[job] = time.time()
                    else:
                        gpu_str = real_gpu + ','
                    V100_job[g] = job
                    new_pool.remove(job)
                start_job(node_string, gpu_str, job)
            started.remove(job)
        # resume promoted jobs
        for job in promoted[:]:
            if finish_dict['job'+job] != 1:
                gpu = mapping[job]
                if job not in multigpu_list:
                    real_node, real_gpu = V100_LUT(gpu)                           
                    resume_job(real_node, real_gpu, job)
                    V100_job[gpu] = job
                else:
                    gpu_split = gpu.split(',')
                    node_string = ''
                    for g in gpu_split:
                        real_node, real_gpu = V100_LUT(g)
                        if g == gpu_split[1]:
                            gpu_str += real_gpu
                            node_string = real_node
                        else:
                            gpu_str = real_gpu + ','
                        V100_job[g] = job
                    resume_job(node_string, gpu_str, job)
                promoted.remove(job)
                num_mig[job] += 1
                V100_used += 1
            else: # job finished before checkpointing
                promoted.remove(job)

        # 2. find all mapping of remaining new jobs (current new_pool list) that are going to start on K80
        # first make sure there are remaining new jobs
        if len(new_pool) > 0:
            K80_avail = get_avail_id(K80_job) 
            K_mapping = K80_placement(K80_avail, new_pool)
            remain_pool = list(set(new_pool).intersection(new_pool)) # just to get rid of duplicated 2-gpu job items
            for job in remain_pool[:]: # new jobs
                gpu = K_mapping[job]
                if job not in multigpu_list:
                    real_node, real_gpu = K80_LUT(gpu)
                    start_job(real_node, real_gpu, job)
                    K80_job[gpu] = job
                    job_start[job] = time.time()
                    queue_delay[job] = int(time.time() - queue_timer - queue_dict[int(job)])
                    K80_start_time[job] = time.time()
                    new_pool.remove(job)
                else:
                    gpu_split = gpu.split(',')
                    node_string = ''
                    for g in gpu_split:
                        real_node, real_gpu = K80_LUT(g)
                        if g == gpu_split[1]:
                            gpu_str += real_gpu
                            node_string = real_node
                            job_start[job] = time.time()
                            queue_delay[job] = int(time.time() - queue_timer - queue_dict[int(job)])
                            K80_start_time[job] = time.time()
                        else:
                            gpu_str = real_gpu + ','
                        K80_job[g] = job
                        new_pool.remove(job)
                    start_job(node_string, gpu_str, job)

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(promoted) > 0 or len(new_pool) > 0:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')

    ############## monitor GPU usage ############

    usage = K80_used + V100_used
    time_stamp = int(time.time() - queue_timer)
    gpu_usage_time.append(time_stamp)
    gpu_usage.append(usage)
    total_completion = np.sum(list(completion.values()))
    gpu_usage_completion.append(total_completion)

    ############### wait for next iteration

    time.sleep(INTERVAL)

    ################ check if termination condition is met ################

    K80_idle_num = sum(value == 'idle' for value in K80_job.values())
    V100_idle_num = sum(value == 'idle' for value in V100_job.values())
    if K80_idle_num == K80_cap and V100_idle_num == V100_cap and index == len(queue):
        print('all jobs are finished!')
        break


# get average JCT
average_JCT = np.average(list(JCT.values()))
JCT['average'] = average_JCT

average_overhead = np.average(list(overhead.values()))
overhead['average'] = average_overhead

average_queue_delay = np.average(list(queue_delay.values()))
queue_delay['average'] = average_queue_delay

# after everything is finished

print('finished all runs')
JCT_name = testcase + '_JCT.json'
overhead_name = testcase + '_overhead.json'
num_mig_name = testcase + '_num_mig.json'
epoch_waste_name = testcase + '_epoch_waste.json'
ckpt_qual_name = 'ckpt_qual.json'
finish_name = 'finish.json'
K80_time_name = testcase + '_K80_time.json'
V100_time_name = testcase + '_V100_time.json'
gpu_usage_name = testcase + '_gpu_usage.csv'
completion_name = 'completion.json'
ovhd_a_name = testcase + '_ovhd_a.json'
ovhd_b_name = testcase + '_ovhd_b.json'
ovhd_c_name = testcase + '_ovhd_c.json'
ovhd_d_name = testcase + '_ovhd_d.json'
ovhd_total_name = testcase + '_ovhd_total.json'
K80_1st_ovhd_name = testcase + '_K80_1st_ovhd.json'
V100_1st_ovhd_name = testcase + '_V100_1st_ovhd.json'
queue_delay_name = testcase + '_queue_delay.json'
K80_batch_time_name = testcase + '_K80_batch_time.json'
V100_batch_time_name = testcase + '_V100_batch_time.json'
birthplace_name = testcase + '_birthplace.json'
speedup_name = testcase + '_speedup.json'
job_remaining_batch_name = 'job_remaining_batch.json'

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
with open(K80_1st_ovhd_name, 'w') as fp3:
    json.dump(K80_1st_ovhd, fp3, sort_keys=True, indent=4)
with open(V100_1st_ovhd_name, 'w') as fp3:
    json.dump(V100_1st_ovhd, fp3, sort_keys=True, indent=4)
with open(completion_name, 'w') as fp1:
   json.dump(completion, fp1, sort_keys=True, indent=4)
with open(queue_delay_name, 'w') as fp1:
   json.dump(queue_delay, fp1, sort_keys=True, indent=4)
with open(K80_batch_time_name, 'w') as fp3:
    json.dump(K80_batch_time, fp3, sort_keys=True, indent=4)
with open(V100_batch_time_name, 'w') as fp3:
    json.dump(V100_batch_time, fp3, sort_keys=True, indent=4)
with open(birthplace_name, 'w') as fp1:
   json.dump(birthplace, fp1, sort_keys=True, indent=4)
with open(speedup_name, 'w') as fp1:
   json.dump(speedup_dict, fp1, sort_keys=True, indent=4)
with open(job_remaining_batch_name, 'w') as fp1:
   json.dump(job_remaining_batch, fp1, sort_keys=True, indent=4)

gpu_usage_time = np.asarray(gpu_usage_time)
gpu_usage = np.asarray(gpu_usage)
gpu_usage_completion = np.asarray(gpu_usage_completion)
rows = zip(gpu_usage_time, gpu_usage, gpu_usage_completion)
with open(gpu_usage_name, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

