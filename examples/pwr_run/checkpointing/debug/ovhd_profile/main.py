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
    queue_dict[item] = 0
queue_timer = time.time()

promo_time = {}
for item in queue:
    promo_time[str(item)] = 0
demo_time = {}
for item in queue:
    demo_time[str(item)] = 0
promo_start = {} # initialize this to 0 as well
for item in queue:
    promo_start[str(item)] = 0
demo_start = {} # initialize this to 0 as well
for item in queue:
    demo_start[str(item)] = 0

queue_start = {} # initialize this to 0 as well
for item in queue:
    queue_start[str(item)] = 0
queue_time = {} # initialize this to 0 as well
for item in queue:
    queue_time[str(item)] = 0

speedup_dict = {}
with open('speedup.json', 'r') as fp:
    speedup_dict = json.load(fp)

index = 0

K80_cap = 1
V100_cap = 1
K80_used = 0
V100_used = 0

K80_job = {}
for i in range(1):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(1):
    V100_job[str(i)] = 'idle'
pc_job = []

K80_node = 'c2180'
V100_node = 'd1020'
host_node = 'c0168'
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
                    global K80_job
                    global v100_job
                    global promo_start
                    global promo_time
                    global demo_start
                    global demo_time
                    global K80_used
                    global V100_used

                    if 'param' in data_str:
                        pass
                    elif 'ckpt_qual' in data_str:
                        global ckpt_qual_dict
                        job_name = data_str.split(' ')[0]
                        ckpt_qual_dict[job_name] = 1
                        job = job_name.replace('job','')
                        # start recording time start
                        pdb.set_trace()
                        if job in list(K80_job.values()) and promo_start[job] == 0:
                            # checkpoint on K80, record promo_start
                            promo_start[job] = time.time()
                            save_job(K80_node, job)
                            print('checkpoint on K80')

                        elif job in list(V100_job.values()) and demo_start[job] == 0:
                            # record promo_time, checkpoint on V100, record demo_start
                            promo_time[job] = int(time.time() - promo_start[job])
                            print('measured promo_time')
                            time.sleep(3)
                            demo_start[job] = time.time()
                            save_job(V100_node, job)
                            print('checkpoint on V100')
                        elif demo_start[job] != 0:
                            # record demo_time, save job, then change K80 back to idle
                            demo_time[job] = int(time.time() - demo_start[job])
                            save_job(K80_node, job)
                            print('measured demo_time')


                    elif 'pid' in data_str:
                        global pid_dict
                        job_name = data_str.split(' ')[0]
                        pid = data_str.split(' ')[2]
                        pid_dict[job_name] = pid
                    elif 'checkpoint' in data_str:
                        job_name = data_str.split(' ')[0]
                        job = job_name.replace('job','')
                        # resume to measure promote time
                        if job in list(K80_job.values()) and job not in list(V100_job.values()):
                            resume_job(V100_node, '0', job) 
                            V100_job[0] = job_new
                            V100_used += 1
                            print('resume on V100')
                        # resume to measure demote time
                        elif job in list(V100_job.values()) and demo_time[job] == 0:
                            resume_job(K80_node, '0', job)
                            print('resume on K80')
                        # clean up everything
                        elif demo_time[job] != 0:
                            V100_job[0] = 'idle'
                            V100_used -= 1
                            K80_job[0] = 'idle'
                            K80_used -= 1
                            time.sleep(3)
                            print('cleaned up')

                    print('received ' + data_str)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()

x = threading.Thread(target=thread_function, daemon=True)
x.start()

pdb.set_trace()
send_signal('c2180', 'abcd')
while True:
    time.sleep(5)

###############################################################################

######################################################################

while True:
    
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
                        K80_job[gpu] = job_new # allocate gpu for it, but don't start yet
                        index += 1
                        K80_used += 1
                        pdb.set_trace()
                        start_job(K80_node, '0', job_new)
                        print('started job' + job_new)
                        break

    ############### wait for next iteration

    time.sleep(INTERVAL)

    ################ check if termination condition is met ################

    K80_idle_num = sum(value == 'idle' for value in K80_job.values())
    V100_idle_num = sum(value == 'idle' for value in V100_job.values())
    if K80_idle_num == K80_cap and V100_idle_num == V100_cap and index == len(queue):
        print('all jobs are finished!')
        break


# after everything is finished

print('finished all runs')
ckpt_qual_name = 'ckpt_qual.json'

with open(ckpt_qual_name, 'w') as fp1:
    json.dump(ckpt_qual_dict, fp1, sort_keys=True, indent=4)
with open('promo_time.json', 'w') as fp1:
    json.dump(promo_time, fp1, sort_keys=True, indent=4)
with open('demo_time.json', 'w') as fp1:
    json.dump(demo_time, fp1, sort_keys=True, indent=4)

