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
index = 0

V100_cap = 4
V100_used = 0
qualified_jobs = 0

V100_job = {}
for i in range(4):
    V100_job[str(i)] = 'idle'
qualified_job = []
pc_job = [] # list of jobs that are pratically completed

V100_node = 'd1020'
host_node = 'c0175'
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

    ################### check for finished jobs on V100 ##############################

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

