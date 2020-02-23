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

parser = argparse.ArgumentParser(description='TCP client')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()

queue = [6, 33, 4, 43, 15, 47, 18, 42, 35, 40, 34, 20, 9, 29, 19, 22, 3, 5, 38, 7, 41, 39, 46, 17, 24, 28, 26, 45, 16, 14, 50, 48, 36, 27, 32, 8, 10, 49, 2, 12, 23, 1, 37, 31, 44, 21, 30, 11, 13, 25] 
queue_dict = {}
arrival_time = 0 
for item in queue:
    arrival_time += np.random.poisson(30)
    queue_dict[item] = arrival_time
queue_timer = time.time()

job_start = {} #{'49': time1, '15': time2...}
JCT = {}
overhead = {} # initialize so that every job starts with 0s overhead time
for item in queue:
    overhead[str(item)] = 0
ovhd_start = {} # initialize this to 0 as well
for item in queue:
    ovhd_start[str(item)] = 0
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
speedup_dict = {}
for item in queue:
    speedup_dict[str(item)] = 1 # initialize to 1, assuming untested jobs have highest speedup

index = 0

K80_cap = 8
V100_cap = 4
K80_used = 0
V100_used = 0

K80_job = {}
for i in range(8):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(4):
    V100_job[str(i)] = 'idle'
step1_job = []
step2_job = []
pc_job = []

K80_node = 'c2180'
V100_node = 'd1020'
host_node = 'c0175'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s

def send_signal(node, cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 10000 if node == K80_node else 10001 
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
                print('received {!r}'.format(data))
                break
            else:
                print('waiting for success signal')
                time.sleep(1)
    finally:
        #print('closing socket')
        sock.close()

def max_speedup_promotion(K80_free, V100_free, V100_job, promote_list, force_demote):
    num_demote = len(force_demote)
    num_promote = len(promote_list)  
    V100_vacant = num_demote + V100_free
    K80_vacant = num_promote + K80_free 
    global speedup_dict
    if K80_vacant >= num_demote: # if more vacant K80s than demote jobs, always demote
        # selectively promote among active V100 jobs and promote list jobs
        V100_qual = list(set(list(V100_job.values())) - set(force_demote))
        if 'idle' in V100_qual:
            V100_qual.remove('idle')
        V100_pool = list(set(V100_qual).union(promote_list))       
        if len(V100_pool) <= 4: # promote all jobs as well
            return promote_list, force_demote
        else: # promote the top 4 jobs            
            pool_dict = {}
            for job in V100_pool:
                if job in speedup_dict:
                    pool_dict[job] = speedup_dict[job]        
            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:4] 
            promotion_list = list(set(promote_list).intersection(sorted_pool))                     
            demotion_list = list(set(list(V100_job.values())).difference(sorted_pool))
            if 'idle' in demotion_list:
                demotion_list.remove('idle') # this includes force demotion
            return promotion_list, demotion_list
    elif V100_vacant >= num_promote: # if more vacant V100s than promote jobs, always promote
        # less vacant K80s than demote jobs, select worst among force demote list
        pool_dict = {} # here the pool only includes force demote jobs
        for job in force_demote:
             if job in speedup_dict:
                pool_dict[job] = speedup_dict[job]
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_vacant] 
        return promote_list, sorted_pool
    else:
        raise ValueError('Bug with max speedup promotion, condition not considered')

def save_job(node, job): # save_job('c2176', '50')
    # first wait for the job to be qualified for checkpointing
    while True: # wait for ckpt_qual to be available
        global ckpt_qual_dict
        if ckpt_qual_dict['job'+job] == 1:
            ckpt_qual_dict['job'+job] = 0
            break
        time.sleep(5)

    send_signal(node, 'save ' + job)

    global ovhd_start
    ovhd_start[job] = time.time() 

    # after sending checkpoint signal, wait for it to finish 
    while True:
        time.sleep(5)
        with open('checkpoint.json', 'r') as fp2:
            checkpoint_dict = json.load(fp2)                        
        if checkpoint_dict['job'+job] == 1: # checkpoint has finished
            print('checkpointed successfully')
            checkpoint_dict['job'+job] = 0 # reset it
            json_file = json.dumps(checkpoint_dict)
            with open('checkpoint.json', 'w') as fp2:
                fp2.write(json_file) 
            break
        # also check if job has already finished
        global finish_dict
        if finish_dict['job'+job] == 1:
            break

# resume job
def resume_job(node, gpu, job): # resume_job('c2176', '3', '50')
    while True:
        if os.path.exists('pid.json'):
            os.rename('pid.json', 'pid_lock.json')
            break
        else:
            time.sleep(1)   

    cmd = 'resume ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

    while True:
        if os.path.exists('pid.json'):
            break
        else:
            time.sleep(1)

# start job
def start_job(node, gpu, job):
    # first wait for pid.json to show up, rename pid.json to pid_lock.json
    # then in jobx.py, modify pid_lock.json, rename it to pid.json
    # then wait for pid.json to show up
    while True:
        if os.path.exists('pid.json'):
            os.rename('pid.json', 'pid_lock.json')
            break
        else:
            time.sleep(1)   

    cmd = 'start ' + job + ' gpu ' + gpu
    send_signal(node, cmd)   

    while True:
        if os.path.exists('pid.json'):
            break
        else:
            time.sleep(1)
           
# function that checks the tensorboard log of currently running jobs and logs jobs that have finished the first epoch
# in a global list. Once it's done, it will be in a queue to be promoted to V100 for 3 more epochs.
def check_step1_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    threshold = 0.01
    global step1_job
    global K80_epoch_time
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
                        K80_epoch_time[job] = wall_time[1] - wall_time[0]
                        step1_job.append(job)
                        print('job' + job + ' has reached step1 complete')
                except Exception:
                    pass

def check_step2_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    threshold = 0.01
    global step1_job
    global step2_job
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
                        K80_time = K80_epoch_time[job]
                        V100_time = wall_time[1] - wall_time[0]
                        speedup = (K80_time - V100_time) / K80_time
                        speedup_dict[job] = speedup
                        step2_job.append(job)
                        print('job' + job + ' has reached step2 complete')
                except Exception:
                    pass


############### first clear finish status of all jobs ####################

pid_dict = {}
with open('pid.json', 'r') as fp:
    pid_dict = json.load(fp)
for key in pid_dict:
    pid_dict[key] = 0
json_file = json.dumps(pid_dict)
with open('pid.json', 'w') as fp:
    fp.write(json_file) 

checkpoint_dict = {}
with open('checkpoint.json', 'r') as fp:
    checkpoint_dict = json.load(fp)
for key in checkpoint_dict:
    checkpoint_dict[key] = 0
json_file = json.dumps(checkpoint_dict)
with open('checkpoint.json', 'w') as fp:
    fp.write(json_file) 

ckpt_qual_dict = {}
for i in range(50):
    job_name = 'job' + str(i + 1)
    ckpt_qual_dict[job_name] = 0

finish_dict = {}
for i in range(50):
    job_name = 'job' + str(i + 1)
    finish_dict[job_name] = 0

epoch_waste_dict = {}
with open('epoch_waste.json', 'r') as fp:
    epoch_waste_dict = json.load(fp)
for key in epoch_waste_dict:
    epoch_waste_dict[key] = 0
json_file = json.dumps(epoch_waste_dict)
with open('epoch_waste.json', 'w') as fp:
    fp.write(json_file) 

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
                    if 'param' in data_str:
                        pass
                    elif 'ckpt_qual' in data_str:
                        global ckpt_qual_dict
                        job_name = data_str.split(' ')[0]
                        ckpt_qual_dict[job_name] = 1
                    elif 'finish' in data_str:
                        global finish_dict
                        job_name = data_str.split(' ')[0]
                        finish_dict[job_name] = 1
                    print('received ' + data_str)
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
                JCT[job] = int(time.time() - job_start[job])  
            elif ovhd_start[job] != 0:
                # check if ckpt overhead has finished
                if ckpt_qual_dict['job'+job] == 1:
                    overhead[job] += int(time.time() - ovhd_start[job])
                    ovhd_start[job] = 0                   

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job)
                JCT[job] = int(time.time() - job_start[job])
            elif ovhd_start[job] != 0:
                # check if ckpt overhead has finished
                if ckpt_qual_dict['job'+job] == 1:
                    overhead[job] += int(time.time() - ovhd_start[job])
                    ovhd_start[job] = 0                   

    ################ check step1 finished job of K80 jobs and step 2 of V100 #################

    check_step1_complete(list(K80_job.values()))
    check_step2_complete(list(V100_job.values()))   

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    K80_free = K80_cap - K80_used
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
    promote_list = list(set(step2_job).intersection(list(K80_job.values())).difference(pc_job))
    # this returns job forced to be demoted. Currently in V100, and is practically complete
    force_promote = list(set(list(K80_job.values())).intersection(step1_job).difference(step2_job))
    promote_list = list(set(promote_list).union(force_promote))
    force_demote = list(set(list(V100_job.values())).intersection(pc_job))

    if len(promote_list) > 0:
        promoted, demoted = max_speedup_promotion(K80_free, V100_free, V100_job, promote_list, force_demote)
        if len(promoted) > 0:
            print('promoted jobs: ', promoted)
        if len(demoted) > 0:
            print('demoted jobs: ', demoted)
        # stop all promoted jobs on K80
        for gpu, job in K80_job.items():
            if job in promoted:
                # make sure promoted step1 job doesn't get demoted back before finishing profiling
                if job in step1_job and job not in step2_job:
                    speedup_dict[job] = 2 
                save_job(K80_node, job)
                K80_job[gpu] = 'idle'
                K80_used -= 1
                
        # stop all demoted jobs on V100
        for gpu, job in V100_job.items():
            if job in demoted:
                save_job(V100_node, job)
                V100_job[gpu] = 'idle'
                V100_used -= 1

        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in promoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in V100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        resume_job(V100_node, gpu, job_new)
                        num_mig[job_new] += 1
                        V100_job[gpu] = job_new
                        promoted.remove(job_new)
                        V100_used += 1
                        break
            else: # job has already finished before checkpointing
                JCT[job_new] = int(time.time() - job_start[job_new])  
                promoted.remove(job_new)
            
        # resume demoted jobs on K80, make sure the gpu is idle
        for job_new in demoted[:]:
            if finish_dict['job'+job_new] != 1:
                for gpu, job in K80_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        resume_job(K80_node, gpu, job_new)
                        num_mig[job_new] += 1
                        K80_job[gpu] = job_new
                        demoted.remove(job_new)
                        K80_used += 1
                        break
            else: # job has already finished before checkpointing
                JCT[job_new] = int(time.time() - job_start[job_new])  
                demoted.remove(job_new)

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(promoted) > 0 or len(demoted) > 0:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')

    ################ submit new jobs to vacant K80 GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
    if K80_used < K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                for gpu, job in K80_job.items():
                    if job == 'idle': # schedule new job here if idle
                        start_job(K80_node, gpu, job_new)
                        K80_job[gpu] = job_new
                        job_start[job_new] = time.time()
                        index += 1
                        K80_used += 1
                        time.sleep(5) # don't communicate too often
                        break

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

# after everything is finished
with open('epoch_waste.json', 'r') as fp:
    epoch_waste_dict = json.load(fp)

print('finished all runs')
JCT_name = testcase + '_JCT.json'
overhead_name = testcase + '_overhead.json'
num_mig_name = testcase + '_num_mig.json'
epoch_waste_name = testcase + '_epoch_waste.json'
ckpt_qual_name = 'ckpt_qual.json'
finish_name = 'finish.json'
speedup_name = 'speedup.json'

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
with open(speedup_name, 'w') as fp1:
   json.dump(speedup_dict, fp1, sort_keys=True, indent=4)

