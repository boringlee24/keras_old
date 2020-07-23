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

parser = argparse.ArgumentParser(description='TCP client')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()

with open('../job_trace/job_queue.json', 'r') as fp:
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

multigpu_list = ['1', '2', '3']

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
v100_1st = {}
for item in queue:
    v100_1st[str(item)] = []

num_mig = {} # initialize migration time to 0
for item in queue:
    num_mig[str(item)] = 0
queue_start = {} # initialize this to 0 as well
for item in queue:
    queue_start[str(item)] = 0
queue_time = {} # initialize this to 0 as well
for item in queue:
    queue_time[str(item)] = 0
K80_epoch_time = {}
for item in queue:
    K80_epoch_time[str(item)] = 0
K80_start_time = {}
for item in queue:
    K80_start_time[str(item)] = 0
V100_start_time = {}
for item in queue:
    V100_start_time[str(item)] = 0
K80_time = {}
for item in queue:
    K80_time[str(item)] = 0
V100_time = {}
for item in queue:
    V100_time[str(item)] = 0
gpu_usage_time = [] # don't initialize this
gpu_usage = []
gpu_usage_completion = []

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
qualified_job = []
pc_job = []

K80_node = ['c2182']
V100_node = ['d1018']
host_node = 'c0176'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 10 # make decision every 30s TODO

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
    indexs = [j for j, e in enumerate(value_list) if e == 'reserved' or e == 'idle']
    return [key_list[j] for j in indexs]

# jobs in K80 and in new pool compete for reserved V100 spots by random chance
# for 2-gpu jobs, the job will have duplicated entry in new_pool and promote_list
# V100_avail is a list with gpuid that is reserved or idle
# random promo does not have job demotion
# 1st return: list of job that starts on V100. 2nd return: list of job that promotes to V100
# 3rd return: dict of mapping {'50':'5', '3':'1,2'} means job50 runs on gpuid 5, job3 runs on gpuid 1,2
def random_promotion(V100_avail, new_pool, promote_list):
    V100_pool = new_pool + promote_list 
    mapping = {}
    ####### this is only used in specific cases ##########
    # group two gpus in same node together
    # [1,2,3,5,8] -> [[1,2],[3],[5,8]]
    skip = False
    res_group = [] # group reserved GPU together
    for i in range(len(V100_avail)):
        if skip:
            skip = False
            continue
        else:
            # two gpus from the same node
            if i!=len(V100_avail)-1 and int(V100_avail[i])//V100_per_node==int(V100_avail[i+1])//V100_per_node:
                skip = True
                res_group.append([V100_avail[i], V100_avail[i+1]])
            else:
                res_group.append([V100_avail[i]])
    group_1gpu = [i for i in res_group if len(i) == 1] # 1gpu id
    group_2gpu = [i for i in res_group if len(i) == 2] # 2gpu id
    pool_1gpu = [i for i in V100_pool if i not in multigpu_list] # 1gpu job
    pool_2gpu = [i for i in V100_pool if i in multigpu_list] # 2gpu job
    ########################################################

    if len(V100_avail) >= len(V100_pool) and len(group_2gpu) >= len(pool_2gpu):
        # this means all jobs get to run on V100
        sorted_pool = V100_pool
        # if there is no 2-gpu job
        if set(V100_pool).isdisjoint(multigpu_list):
            for i in range(len(sorted_pool)):
                mapping[sorted_pool[i]] = V100_avail[i]
        # there are 2-gpu jobs
        else:
            # first, fill in all 1gpu slots with 1-gpu jobs as much as possible
            for i in group_1gpu:
                if len(pool_1gpu) > 0:
                    mapping[pool_1gpu[0]] = i
                    pool_1gpu.pop(0)
            for i in group_2gpu:
                if len(pool_2gpu) > 1:
                    mapping[pool_2gpu[0]] = ','.join(i)
                    pool_2gpu = [i for i in pool_2gpu if i != pool_2gpu[0]]
                elif len(pool_1gpu) > 0:
                    mapping[pool_1gpu[0]] = i[0]
                    if len(pool_1gpu) > 1:
                        mapping[pool_1gpu[1]] = i[1]
                        pool_1gpu.pop(1)
                    pool_1gpu.pop(0)
    else:
        # if there are no 2-gpu jobs at all
        if set(V100_pool).isdisjoint(multigpu_list):
            sorted_pool = random.sample(V100_pool, len(V100_avail))
            for i in range(len(sorted_pool)):
                mapping[sorted_pool[i]] = V100_avail[i]
        # if there are 2-gpu jobs but no reserved spots for it
        elif len(group_2gpu) == 0:
            # remove 2-gpu jobs from V100_pool
            V100_pool = [i for i in V100_pool if i not in multigpu_list]
            num_sample = min(len(V100_avail), len(V100_pool)) # in case jobs are less than slots after reduction
            sorted_pool = random.sample(V100_pool, num_sample)
            for i in range(len(sorted_pool)):
                mapping[sorted_pool[i]] = V100_avail[i]
        # if there are 2-gpu jobs with available spots
        else:
            print('there are 2-gpu jobs with available 2-gpu slots')
            print('V100 pool:', V100_pool)
            sorted_pool = []
            for i in group_1gpu:
                if len(pool_1gpu) > 0:
                    picked = random.choice(pool_1gpu)
                    sorted_pool.append(picked)
                    pool_1gpu.remove(picked)
                    V100_pool.remove(picked)
                    mapping[picked] = i
            for i in group_2gpu:
                picked = random.choice(V100_pool)
                if picked in pool_2gpu:
                    sorted_pool.append(picked)
                    sorted_pool.append(picked)
                    V100_pool = [i for i in V100_pool if i != picked]
                    pool_2gpu = [i for i in pool_2gpu if i != picked]
                    mapping[picked] = ','.join(i)
                else:
                    # pick another 1-gpu job to fill in the 2-gpu slot
                    sorted_pool.append(picked)
                    pool_1gpu.remove(picked)
                    V100_pool.remove(picked)
                    mapping[picked] = i[0]
                    picked_2 = random.choice(pool_1gpu)
                    sorted_pool = append(picked_2)
                    pool_1gpu.remove(picked_2)
                    V100_pool.remove(picked_2)
                    mapping[picked_2] = i[1]
            print('picked jobs:', sorted_pool)
    start_list = list(set(sorted_pool).intersection(new_pool))
    promo_list = list(set(sorted_pool).intersection(promote_list))
    return start_list, promo_list, mapping

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
pdb.set_trace()
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
                    global ovhd_a, ovhd_b, ovhd_c, ovhd_d, k80_1st, v100_1st, ovhd_start, overhead, ovhd_total
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
                    elif '1st_epoch' in data_str: # 'job50 1st_epoch 35'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        epoch_time = int(data_str.split(' ')[2])
                        if job in list(K80_job.values()):
                            k80_1st[job].append(epoch_time)
                        elif job in list(V100_job.values()):
                            v100_1st[job].append(epoch_time)
                    elif 'completion' in data_str: # 'job50 completion 0.33'
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        completion_portion = float(data_str.split(' ')[2])
                        completion[job] = completion_portion
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

    ################ 

    ################ submit new jobs to vacant K80 GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
    # here, just check how many new jobs can get started on idle GPUs and put them in new_pool[]. 
    # But do not allocate any GPU for the jobs yet.
    new_pool = []
    if V100_used < V100_cap:
        V100_free = V100_cap - V100_used
        for i in range(V100_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                if job_new in multigpu_list:
                    idle_gpus = detect_2_gpus(V100_job, V100_per_node)[:2]
                    if len(idle_gpus) > 0:
                        index += 1
                        qualified_job.append(job_new)
                        # new_pool will have duplicated job for 2-gpu ones
                        for gpu in idle_gpus:
                            new_pool.append(job_new) 
                            V100_job[gpu] = 'reserved' 
                            V100_used += 1 
                else:
                    for gpu, job in V100_job.items():
                        if job == 'idle': # schedule new job here if idle
                            new_pool.append(job_new)
                            qualified_job.append(job_new)
                            V100_job[gpu] = 'reserved'
                            index += 1
                            V100_used += 1
                            break

    if K80_used < K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                if job_new in multigpu_list:
                    idle_gpus = detect_2_gpus(K80_job, K80_per_node)[:2]
                    if len(idle_gpus) > 0:
                        index += 1
                        qualified_job.append(job_new)
                        for gpu in idle_gpus:
                            new_pool.append(job_new) 
                            K80_job[gpu] = 'reserved' 
                            K80_used += 1 
                else:
                    for gpu, job in K80_job.items():
                        if job == 'idle': # schedule new job here if idle
                            new_pool.append(job_new)
                            qualified_job.append(job_new)
                            K80_job[gpu] = 'reserved'
                            index += 1
                            K80_used += 1
                            break

    # make promotion decisions
    pdb.set_trace()
    V100_avail = get_status_id(V100_job)
    promote_list = [i for i in list(K80_job.values()) if i in qualified_job]

    if len(promote_list) > 0 or len(new_pool) > 0:
        # started and promoted do not have duplicated elements
        started, promoted = random_promotion(V100_avail, new_pool, promote_list)
        if len(started) > 0:
            print('jobs starting on V100: ', started)
        if len(promoted) > 0:
            print('promoted jobs: ', promoted)
        # stop all promoted jobs on K80
        checkpoint_finish_check = []
        for gpu, job in K80_job.items():
            if job in promoted:
                if job not in new_pool: # don't do checkpointing for new jobs
                    real_node, real_gpu = K80_LUT(gpu)
                    save_job(real_node, job)
                    if finish_dict['job'+job] != 1:
                        K80_time[job] += int(time.time() - K80_start_time[job])
                    checkpoint_finish_check.append(job)
                K80_job[gpu] = 'idle'
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

        # 1. deal with all V100 jobs (started, promoted). Start from 2-gpu jobs if there are any
        started_2gpu = [i for i in started if i in multigpu_list]
        started_1gpu = [i for i in started if i not in multigpu_list]

        




        V100_new_remain = list(set(demote_list).difference(demoted))
        # start remaining new jobs on K80, make sure the gpu equals its allocated one
        for job_new in V100_new_remain:
            for gpu, job in V100_job.items():
                if job == job_new: # if gpu idle, schedule new job here
                    real_node, real_gpu = V100_LUT(gpu)
                    start_job(real_node, real_gpu, job_new)
                    job_start[job_new] = time.time()
                    queue_delay[job_new] = int(time.time() - queue_timer - queue_dict[int(job_new)])
                    V100_start_time[job_new] = time.time()
                    new_pool.remove(job_new)
                    break

        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in promoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in V100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        if job_new in new_pool:
                            real_node, real_gpu = V100_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time.time() - queue_timer - queue_dict[int(job_new)])
                            V100_start_time[job_new] = time.time()
                            new_pool.remove(job_new)
                        else:
                            real_node, real_gpu = V100_LUT(gpu)                           
                            resume_job(real_node, real_gpu, job_new)
                            num_mig[job_new] += 1
                        V100_job[gpu] = job_new
                        promoted.remove(job_new)
                        V100_used += 1
                        break
                        
            else: # job has already finished before checkpointing
                promoted.remove(job_new)

        # start remaining new jobs on K80, make sure the gpu equals its allocated one
        for job_new in new_pool[:]:
            for gpu, job in K80_job.items():
                if job == job_new: # if gpu idle, schedule new job here
                    real_node, real_gpu = K80_LUT(gpu)
                    start_job(real_node, real_gpu, job_new)
                    job_start[job_new] = time.time()
                    queue_delay[job_new] = int(time.time() - queue_timer - queue_dict[int(job_new)])
                    K80_start_time[job_new] = time.time()
                    new_pool.remove(job_new)
                    break

        # resume demoted jobs on K80, make sure the gpu is idle
        for job_new in demoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in K80_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        real_node, real_gpu = K80_LUT(gpu)
                        start_job(real_node, real_gpu, job_new)
                        job_start[job_new] = time.time()
                        queue_delay[job_new] = int(time.time() - queue_timer - queue_dict[int(job_new)])
                        K80_start_time[job_new] = time.time()
                        new_pool.remove(job_new)
                        K80_job[gpu] = job_new
                        demoted.remove(job_new)
                        K80_used += 1
                        break
            else: # job has already finished before checkpointing
                demoted.remove(job_new)

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(promoted) > 0 or len(demoted) > 0 or len(new_pool) > 0:
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
k80_1st_name = testcase + '_k80_1st.json'
v100_1st_name = testcase + '_v100_1st.json'
queue_delay_name = testcase + '_queue_delay.json'

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
with open(k80_1st_name, 'w') as fp3:
    json.dump(k80_1st, fp3, sort_keys=True, indent=4)
with open(v100_1st_name, 'w') as fp3:
    json.dump(v100_1st, fp3, sort_keys=True, indent=4)
with open(completion_name, 'w') as fp1:
   json.dump(completion, fp1, sort_keys=True, indent=4)
with open(queue_delay_name, 'w') as fp1:
   json.dump(queue_delay, fp1, sort_keys=True, indent=4)

gpu_usage_time = np.asarray(gpu_usage_time)
gpu_usage = np.asarray(gpu_usage)
gpu_usage_completion = np.asarray(gpu_usage_completion)
rows = zip(gpu_usage_time, gpu_usage, gpu_usage_completion)
with open(gpu_usage_name, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

