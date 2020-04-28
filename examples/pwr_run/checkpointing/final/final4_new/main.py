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
V100_epoch_time = {}
for item in queue:
    V100_epoch_time[str(item)] = 0
K80_epoch_time = {}
for item in queue:
    K80_epoch_time[str(item)] = 0

K80_start_time = {}
for item in queue:
    K80_start_time[str(item)] = 0
V100_start_time = {}
for item in queue:
    V100_start_time[str(item)] = 0

promote_start_time = {}
for item in queue:
    promote_start_time[str(item)] = 0
demote_list = []

K80_time = {}
for item in queue:
    K80_time[str(item)] = 0
V100_time = {}
for item in queue:
    V100_time[str(item)] = 0
gpu_usage_time = [] # don't initialize this
gpu_usage = []
gpu_usage_completion = []

speedup_dict = {}
for item in queue:
    speedup_dict[str(item)] = 0 
predict_dict = {}
for item in queue:
    predict_dict[str(item)] = 0 

index = 0
all_jobs_started = False

K80_cap = 16
V100_cap = 8
K80_used = 0
V100_used = 0

K80_job = {}
for i in range(K80_cap):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(V100_cap):
    V100_job[str(i)] = 'idle'
qualified_job = []
step1_job = []
step2_job = []
pc_job = []

K80_node = ['c2180', 'c2181']
V100_node = ['d1018', 'd1012']
host_node = 'c0185'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s

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

######################### do a regression fit ########################
with open('x1_data.json') as f:
    x1_data = json.load(f)
with open('x2_data.json') as f:
    x2_data = json.load(f)
with open('x3_data.json') as f:
    x3_data = json.load(f)

x1_norm = [(i - min(x1_data)) / (max(x1_data) - min(x1_data)) for i in x1_data]
x2_norm = [(i - min(x2_data)) / (max(x2_data) - min(x2_data)) for i in x2_data]
x3_norm = [(i - min(x3_data)) / (max(x3_data) - min(x3_data)) for i in x3_data]

# create training data
x_train = []
for i in range(len(x1_norm)):
    x_train.append([x1_norm[i], x2_norm[i], x3_norm[i]])

with open('y_data.json') as f:
    y_train = json.load(f)

model = neighbors.KNeighborsRegressor(n_neighbors = 3, weights='distance')
model.fit(x_train, y_train)

####################################################################

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

def max_speedup_promotion(K80_free, V100_free, V100_job, promote_list, demote_list, force_demote):
    num_demote = len(force_demote)
    num_promote = len(promote_list)  
    V100_vacant = num_demote + V100_free
    K80_vacant = num_promote + K80_free 
    global speedup_dict
    if K80_vacant >= num_demote: # if more vacant K80s than demote jobs, always force demote
        # selectively promote among active V100 jobs and promote list jobs
        V100_qual = demote_list
        #if 'idle' in V100_qual:
        #    V100_qual.remove('idle')
        V100_pool = list(set(V100_qual).union(promote_list))       
        if num_promote <= V100_vacant: # promote all jobs as well
            return promote_list, force_demote
        else: # promote the top 4 jobs            
            pool_dict = {}
            V100_avail = V100_vacant + len(V100_qual)
            for job in V100_pool:
                if job in speedup_dict:
                    pool_dict[job] = speedup_dict[job]        
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
                            if speedup_dict[job_promote] - speedup_dict[job_demote] < 0.05:
                                demotion_list.remove(job_demote)
                                promotion_list.remove(job_promote)
                                break
            return promotion_list, demotion_list
    # situations below won't happen
    elif V100_vacant >= num_promote: # if more vacant V100s than promote jobs, always promote
        # less vacant K80s than demote jobs, select worst among force demote list
        pool_dict = {} # here the pool only includes force demote jobs
        for job in force_demote:
             if job in speedup_dict:
                pool_dict[job] = speedup_dict[job]
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_vacant]
        if len(sorted_pool) > 0:
            raise ValueError('Bug, demotion shouldnt happen because no practical complete')           
        return promote_list, sorted_pool
    else:
        raise ValueError('Bug with max speedup promotion, condition not considered')

def min_speedup_demotion(K80_job, demote_list):
    num_demote = len(demote_list)
    global speedup_dict

    # selectively demote among active K80 jobs and demote list jobs
    K80_qual = list(set(list(K80_job.values())))
    if 'idle' in K80_qual:
        K80_qual.remove('idle')
    K80_pool = list(set(K80_qual).union(demote_list))       
    if len(K80_pool) <= K80_cap: # demote all jobs, no promotion
        return [], demote_list[:] # must return a copy, otherwise the output points to the same address as input
    else: # promote the top 4 jobs            
        pool_dict = {}
        for job in K80_pool:
            if job in speedup_dict:
                pool_dict[job] = speedup_dict[job]        
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_cap] # 8 least speedup jobs
        demotion_list = list(set(demote_list).intersection(sorted_pool))
        promotion_list = list(set(list(K80_job.values())).difference(sorted_pool))
        if 'idle' in promotion_list:
            promotion_list.remove('idle') # this includes force demotion
        # lazy migration, for every V100 job from high speeup to low speedup and not in sorted_pool, compare it with
        # K80 jobs in sorted_pool, from low speedup to high speedup. If difference within 0.2, replace the K80 job
        # in sorted pool
        for job_demote in sorted(pool_dict, key=pool_dict.get, reverse=True):
            if job_demote in demotion_list:
                for job_promote in sorted(pool_dict, key=pool_dict.get, reverse=False):
                    if job_promote in promotion_list:
                        if speedup_dict[job_promote] - speedup_dict[job_demote] < 0.05:
                            demotion_list.remove(job_demote)
                            promotion_list.remove(job_promote)
                            break

        return promotion_list, demotion_list

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
def check_step1_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    global step1_job
    global V100_epoch_time
    for job in job_list:
        if job not in step1_job and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            if len(dirs) > 0:
                tc = dirs[0]
                iterator = EventAccumulator(tc).Reload()
                tag = 'loss'
                try:
                    if len(iterator.Scalars(tag)) > 2: # this way we can collect one epoch time
                        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
                        V100_epoch_time[job] = wall_time[1] - wall_time[0]
                        step1_job.append(job)
                        print('job' + job + ' has reached step1 complete')
                except Exception:
                    pass

def check_step2_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    global step1_job
    global step2_job
    global V100_epoch_time
    global K80_epoch_time
    global speedup_dict

    for job in job_list:
        if job in step1_job and job not in step2_job and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            if len(dirs) > 1:
                tc = dirs[1]
                iterator = EventAccumulator(tc).Reload()
                tag = 'loss'
                try:
                    if len(iterator.Scalars(tag)) > 2: # this way we can collect one epoch time
                        wall_time = [t.wall_time for t in iterator.Scalars(tag)]
                        K80_epoch_time[job] = wall_time[1] - wall_time[0]
                        V100_time_step2 = V100_epoch_time[job]
                        K80_time_step2 = wall_time[1] - wall_time[0]
                        speedup = (K80_time_step2 - V100_time_step2) / K80_time_step2
                        speedup_dict[job] = speedup
                        step2_job.append(job)
                        print('job' + job + ' has reached step2 complete')
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
                    global V100_start_time, promote_start_time
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
                                promote_start_time[job] = time.time()
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
                print('K80 finished job: ' + job)

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job)
                if job in demote_list:
                    demote_list.remove(job)

    ################ check step1 finished job of K80 jobs and step 2 of V100 #################

    check_step1_complete(list(V100_job.values()))
    check_step2_complete(list(K80_job.values()))   

    for gpu, job in V100_job.items():
        if job not in qualified_job and job != 'idle':
            if job in step1_job:
                real_node, real_gpu = V100_LUT(gpu)
                kill_job(real_node, job)
                qualified_job.append(job)
                print('job' + job + ' has been qualified for demotion')
                time.sleep(3) # wait for run.sh to finish
                x1, x3 = gpu_pwr.process_csv('job'+job, testcase)
                x2 = 3600 / V100_epoch_time[job]
                # preprocess the data
                x1 = (x1 - min(x1_data)) / (max(x1_data) - min(x1_data))
                x2 = (x2 - min(x2_data)) / (max(x2_data) - min(x2_data))
                x3 = (x3 - min(x3_data)) / (max(x3_data) - min(x3_data))

                speedup_pred = model.predict(np.array([x1, x2, x3]).reshape((1,-1)))[0] / 100
                speedup_dict[job] = speedup_pred
                predict_dict[job] = speedup_pred

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
    K80_free = K80_cap - K80_used
    if new_arrival == 0:
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
        promote_list = list(set(qualified_job).intersection(list(K80_job.values())).difference(pc_job))
    else:
        promote_list = []

    # this returns job forced to be demoted. Currently in V100, and is practically complete
    force_demote = list(set(list(V100_job.values())).intersection(pc_job))

    # look at demote list
    for gpu, job in V100_job.items():
        if job != 'idle':
            if job not in demote_list and job in step2_job and len(ovhd_total[job]) > 0:
                job_speedup = speedup_dict[job] # 0.7
                job_ovhd = np.mean(ovhd_total[job]) # 100
                k80_1st_ovhd = np.mean(k80_1st[job]) - K80_epoch_time[job]
                v100_1st_ovhd = np.mean(v100_1st[job]) - V100_epoch_time[job]
                demote_qualify_time = (2 * job_ovhd + k80_1st_ovhd + v100_1st_ovhd) / job_speedup
                if int(time.time() - promote_start_time[job]) > max(demote_qualify_time, max(v100_1st[job])):
                    demote_list.append(job)
                    print('job' + job + 'qualified for demote for passing demote qualify time ' +
                    str(int(demote_qualify_time)))
            elif job not in demote_list and job not in step2_job and job in qualified_job:
                demote_list.append(job)
                print('job' + job + 'qualified for demote for profiling')

    if len(promote_list) > 0 or len(demote_list) > 0:
        if new_arrival == 0:
            promoted, demoted = max_speedup_promotion(K80_free, V100_free, V100_job, promote_list, demote_list, force_demote)
        else:
            promoted, demoted = min_speedup_demotion(K80_job, demote_list)
            if len(demoted) - len(promoted) > new_arrival - V100_free:
                # demote only # of new arrivals + # of promoted
                print('some demoted canceled because more demoted than new arrival + promoted, arrival = ' +
                str(new_arrival))
                print('original demotion: ' + str(demoted))
                demoted_pool = {}
                for job in demoted:
                    if job in speedup_dict:
                        demoted_pool[job] = speedup_dict[job]
                if len(promoted) + new_arrival - V100_free > 0:
                    demoted = sorted(demoted_pool, key=demoted_pool.get, reverse=False)[:(len(promoted)+new_arrival-V100_free)]
                else:
                    demoted = []
                print('new demotion: ' + str(demoted))

        if len(promoted) > 0:
            if new_arrival == 0:
                print('no new job arrivals')
            print('promoted jobs: ', promoted)
        if len(demoted) > 0:
            print('demoted jobs: ', demoted)
        # stop all promoted jobs on K80
        checkpoint_finish_check = []
        for gpu, job in K80_job.items():
            if job in promoted:
                real_node, real_gpu = K80_LUT(gpu)
                save_job(real_node, job)
                if finish_dict['job'+job] != 1:
                    K80_time[job] += int(time.time() - K80_start_time[job])
                checkpoint_finish_check.append(job)
                K80_job[gpu] = 'idle'
                K80_used -= 1
                
        # stop all demoted jobs on V100
        for gpu, job in V100_job.items():
            if job in demoted:
                # make sure demoted step1 job doesn't get promoted back before finishing profiling
                if job in step1_job and job not in step2_job:
                    speedup_dict[job] = 0.01 
                real_node, real_gpu = V100_LUT(gpu)
                save_job(real_node, job)
                if finish_dict['job'+job] != 1:
                    V100_time[job] += int(time.time() - V100_start_time[job])
                checkpoint_finish_check.append(job)
                V100_job[gpu] = 'idle'
                V100_used -= 1
                demote_list.remove(job)

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

        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in promoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in V100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        V100_job[gpu] = job_new
                        real_node, real_gpu = V100_LUT(gpu)                           
                        resume_job(real_node, real_gpu, job_new)
                        num_mig[job_new] += 1
                        promoted.remove(job_new)
                        V100_used += 1
                        break
            else: # job has already finished before checkpointing
                promoted.remove(job_new)

        # resume demoted jobs on K80, make sure the gpu is idle
        for job_new in demoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in K80_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        real_node, real_gpu = K80_LUT(gpu)
                        resume_job(real_node, real_gpu, job_new)
                        num_mig[job_new] += 1
                        K80_job[gpu] = job_new
                        demoted.remove(job_new)
                        K80_used += 1
                        break
            else: # job has already finished before checkpointing
                print('job'+job_new+' has finished before checkpointing')
                demoted.remove(job_new)

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(promoted) > 0 or len(demoted) > 0:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')

    ################ submit new jobs to vacant K80 GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
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
ovhd_a_name = testcase + '_ovhd_a.json'
ovhd_b_name = testcase + '_ovhd_b.json'
ovhd_c_name = testcase + '_ovhd_c.json'
ovhd_d_name = testcase + '_ovhd_d.json'
ovhd_total_name = testcase + '_ovhd_total.json'
k80_1st_name = testcase + '_k80_1st.json'
v100_1st_name = testcase + '_v100_1st.json'
speedup_name = 'speedup.json'
predict_name = 'predict.json'
demote_list_name = 'demote_list.json'
completion_name = 'completion.json'
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
with open(speedup_name, 'w') as fp1:
   json.dump(speedup_dict, fp1, sort_keys=True, indent=4)
with open(predict_name, 'w') as fp1:
   json.dump(predict_dict, fp1, sort_keys=True, indent=4)
with open(demote_list_name, 'w') as fp1:
   json.dump(demote_list, fp1, sort_keys=True, indent=4)
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

