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

queue = []
for i in range(50):
    queue.append(i+1)

queue_dict = {}
arrival_time = 0 
for item in queue:
    arrival_time += np.random.poisson(30)
    queue_dict[item] = arrival_time
queue_timer = time.time()

job_start = {} #{'49': time1, '15': time2...}
JCT = {}
for item in queue:
    JCT[str(item)] = 0
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
index = 0

V100_cap = 4
V100_used = 0
qualified_jobs = 0

V100_job = {}
for i in range(4):
    V100_job[str(i)] = 'idle'
qualified_job = []
pc_job = [] # list of jobs that are pratically completed

V100_node = 'd1021'
host_node = 'c0192'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s
QUALIFY_TIME = 300 # 600s or 10min as threshold

def send_signal(node, cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 10001 if node == V100_node else 10001 
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
for i in range(50):
    job_name = 'job' + str(i + 1)
    pid_dict[job_name] = 0

ckpt_qual_dict = {}
for i in range(50):
    job_name = 'job' + str(i + 1)
    ckpt_qual_dict[job_name] = 0

finish_dict = {}
for i in range(50):
    job_name = 'job' + str(i + 1)
    finish_dict[job_name] = 0

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
                        # move overhead profiling here
                        global ovhd_start
                        global overhead
                        job = job_name.replace('job','')
                        if ovhd_start[job] != 0:
                            if ckpt_qual_dict[job_name] == 1:
                                overhead[job] += int(time.time() - ovhd_start[job])
                                ovhd_start[job] = 0                   
                    elif 'finish' in data_str:
                        global finish_dict
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job', '')
                        finish_dict[job_name] = 1
                        JCT[job] = int(time.time() - job_start[job])  
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

    ################### check for finished jobs on V100 ##############################

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job)

    ################ check run time of current V100 job, update qualified_job #################

    for job in list(V100_job.values()):
        if job not in qualified_job and job != 'idle':
            runtime = int(time.time() - job_start[job])
            if runtime >= QUALIFY_TIME:
                qualified_job.append(job)
                print('job' + job + ' has been qualified for promotion')

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used

    ################ submit new jobs to vacant V100 GPUs ############################

    # check if there are vacant V100s
    ## yes: submit jobs from queue
    ## no: do nothing
    if V100_used < V100_cap:
        V100_free = V100_cap - V100_used
        for i in range(V100_free):
            time_passed = int(time.time() - queue_timer)
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                for gpu, job in V100_job.items():
                    if job == 'idle': # schedule new job here if idle
                        start_job(V100_node, gpu, job_new)
                        V100_job[gpu] = job_new
                        job_start[job_new] = time.time()
                        index += 1
                        V100_used += 1
                        time.sleep(5) # don't communicate too often
                        break

    ############### wait for next iteration

    time.sleep(INTERVAL)

    ################ check if termination condition is met ################

    V100_idle_num = sum(value == 'idle' for value in V100_job.values())
    if V100_idle_num == V100_cap and index == len(queue):
        print('all jobs are finished!')
        break


# get average JCT
average_JCT = np.average(list(JCT.values()))
JCT['average'] = average_JCT

average_overhead = np.average(list(overhead.values()))
overhead['average'] = average_overhead

print('finished all runs')
JCT_name = testcase + '_JCT.json'
overhead_name = testcase + '_overhead.json'
num_mig_name = testcase + '_num_mig.json'
ckpt_qual_name = 'ckpt_qual.json'
finish_name = 'finish.json'

with open(JCT_name, 'w') as fp1:
    json.dump(JCT, fp1, sort_keys=True, indent=4)
with open(overhead_name, 'w') as fp3:
    json.dump(overhead, fp3, sort_keys=True, indent=4)
with open(num_mig_name, 'w') as fp3:
    json.dump(num_mig, fp3, sort_keys=True, indent=4)
with open(ckpt_qual_name, 'w') as fp1:
    json.dump(ckpt_qual_dict, fp1, sort_keys=True, indent=4)
with open(finish_name, 'w') as fp1:
    json.dump(finish_dict, fp1, sort_keys=True, indent=4)

