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

with open('../job_trace/job_queue_100.json', 'r') as fp:
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

multigpu_list = ['1', '2', '3', '4', '5', '6', '7']

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

speedup_dict = {}
with open('speedup.json', 'r') as fp:
    speedup_dict = json.load(fp)

index = 0

K80_cap = 16
V100_cap = 8
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

K80_node = ['c2178', 'c2182']
V100_node = ['d1014', 'd1015']
host_node = 'c0145'
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

# function to detect if there are two free GPUs in a node
# returns an empty list if there is none, otherwise returns list with gpu id in V100/K80_jobs
def detect_2_gpus(gpu_dict, gpu_per_node):
    job_list = list(gpu_dict.values())
    num_nodes = int(len(job_list) / gpu_per_node)
    for i in range(num_nodes):
        start = i * gpu_per_node
        end = i + gpu_per_node
        sliced_list = job_list[start:end]
        occurence = sliced_list.count('idle')
        if occurence >= 2:
            # only take the first two elements
            indexs = [j for j, e in enumerate(sliced_list) if e == 'idle'][:2]
            return [str(j + start) for j in indexs]
    return []

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
                    global completion
                    if 'param' in data_str:
                        pass
                    elif 'ckpt_qual' in data_str:
                        global ckpt_qual_dict
                        job_name = data_str.split(' ')[0]
                        ckpt_qual_dict[job_name] = 1
                        # move overhead profiling here
                        global ovhd_start
                        global overhead
                        job = job_name.replace('job','')
                        if ovhd_start[job] != 0:
                            overhead[job] += int(time.time() - ovhd_start[job])
                            ovhd_start[job] = 0 
                            if job in list(K80_job.values()):
                                K80_start_time[job] = time.time()
                            elif job in list(V100_job.values()):
                                V100_start_time[job] = time.time()
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
                    elif 'checkpoint' in data_str:
                        global checkpoint_dict
                        job_name = data_str.split(' ')[0]
                        checkpoint_dict[job_name] = 1
                    elif 'waste' in data_str:
                        global epoch_waste_dict
                        job_name = data_str.split(' ')[0]
                        epoch_waste_time = data_str.split(' ')[2]
                        epoch_waste_dict[job_name] += int(epoch_waste_time)
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
                        time.sleep(5) # don't communicate too often
                else:
                    for gpu, job in V100_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = V100_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
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
                        time.sleep(5) # don't communicate too often
                else:
                    for gpu, job in K80_job.items():
                        if job == 'idle': # schedule new job here if idle
                            real_node, real_gpu = K80_LUT(gpu)
                            start_job(real_node, real_gpu, job_new)
                            K80_job[gpu] = job_new
                            job_start[job_new] = time.time()
                            queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                            K80_start_time[job_new] = time.time()
                            index += 1
                            K80_used += 1
                            time.sleep(5) # don't communicate too often
                            break

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

    if int(time.time() - queue_timer) > 36000:
        pdb.set_trace()


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


