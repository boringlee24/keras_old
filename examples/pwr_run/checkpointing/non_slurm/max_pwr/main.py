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

parser = argparse.ArgumentParser(description='TCP client')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()

queue = [49, 15, 50, 39, 14, 40, 13, 37, 32, 44, 1, 25, 6, 12, 43, 35, 29, 7, 46, 23, 47, 34, 21, 33, 36, 24, 28, 48, 17, 8, 45, 30, 2, 41, 16, 3, 27, 20, 38, 11, 42, 10, 22, 4, 18, 19, 5, 9, 26, 31]
job_start = {} #{'49': time1, '15': time2...}
JCT = {}
PJCT = {} # practical complete time, not applicable for all jobs
index = 0

K80_cap = 8
V100_cap = 4
K80_used = 0
V100_used = 0
qualified_jobs = 0

K80_job = {}
for i in range(8):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(4):
    V100_job[str(i)] = 'idle'
all_job = []
qualified_job = []
pc_job = [] # list of jobs that are pratically completed

K80_node = 'c2178'
V100_node = 'd1004'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s
QUALIFY_TIME = 600 # 600s or 10min as threshold

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

def max_power_promotion(K80_free, V100_free, V100_job, promote_list, force_demote):
    num_demote = len(force_demote)
    num_promote = len(promote_list)  
    V100_vacant = num_demote + V100_free
    K80_vacant = num_promote + K80_free 
    power_dict = {}
    with open('power.json', 'r') as fp:
        power_dict = json.load(fp)
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
                if 'job'+job in power_dict:
                    pool_dict[job] = power_dict['job'+job]        
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
             if 'job'+job in power_dict:
                pool_dict[job] = power_dict['job'+job]
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_vacant] 
        return promote_list, sorted_pool
    else:
        raise ValueError('Bug with max power promotion, condition not considered')

#a, b = max_power_promotion(1, 1, {0: '49', 1: '39', 2: '50', 3: 'idle'}, ['40', '37'], []) 
#c, d = max_power_promotion(1, 1, {0: 'idle', 1: 'idle', 2: 'idle', 3: 'idle'}, [], []) 

def save_job(node, job): # save_job('c2176', '50')
    # first wait for the job to be qualified for checkpointing
    while True: # wait for ckpt_qual.json to be available
        if os.path.exists('ckpt_qual.json'):
            with open('ckpt_qual.json', 'r') as fp2:
                ckpt_qual_dict = json.load(fp2)
            if ckpt_qual_dict['job'+job] == 1:
                if os.path.exists('ckpt_qual.json'): #if not locked. lock it and edit
                    os.rename('ckpt_qual.json', 'ckpt_qual_lock.json') # lock
                    with open('ckpt_qual_lock.json', 'r') as fp2:
                        ckpt_qual_lock_dict = json.load(fp2)
                    ckpt_qual_lock_dict['job'+job] = 0 # reset it
                    json_file = json.dumps(ckpt_qual_lock_dict)
                    with open('ckpt_qual_lock.json', 'w') as fp2:
                        fp2.write(json_file)
                    os.rename('ckpt_qual_lock.json', 'ckpt_qual.json')
                    break
        time.sleep(5)
    # after passing the ready check, send signal to checkpoint
    send_signal(node, 'save ' + job)

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
        with open('finish.json', 'r') as fp3:
            finish_dict = json.load(fp3)
        if finish_dict['job'+job] == 1:
            break

def kill_job(node, job): # kill_job('c2176', '50')
    send_signal(node, 'kill ' + job)

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

# measure job
def measure_job(node, gpu, job):
    cmd = 'measure ' + job + ' gpu ' + gpu
    send_signal(node, cmd)

# function that checks the tensorboard log of currently running jobs and logs practical complete jobs in a global list
# once a job reaches practical complete, it cannot be promoted. If it's already promoted, it gets demoted.
# criteria for practical complete: loss improvement has been smaller than 0.01 for last 3 consecutive epochs
def check_practical_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    threshold = 0.001
    global pc_job
    global PJCT
    for job in job_list:
        # only check for job outside of practical complete job list
        if job not in pc_job and job != 'idle':
            log_dir = log_path + 'job' + job + '/*'
            dirs = glob.glob(log_dir)
            dirs.sort()
            loss_combine = []
            for tc in dirs:
                
                iterator = EventAccumulator(tc).Reload()
                if len(iterator.Tags()['scalars']) > 0:
                    tag = 'loss' #iterator.Tags()['scalars'][2] # this is tag for loss
                    loss = [item.value for item in iterator.Scalars(tag)]
                    loss_combine += loss

            # now that we have the loss at each epoch, we can check if it has reached practical complete
            if len(loss_combine) >= 4:
                latest_loss = loss_combine[-4:]
                finished = True
                for i in range(3):
                    # if the difference is >= 0.01, the job has not reached practical complete yet
                    if latest_loss[i] - latest_loss[i+1] >= threshold:
                        finished = False
                        break
                if finished:
                    print('job' + job + ' has reached practical complete, the last 4 loss values are')
                    print(str(latest_loss))
                    pc_job.append(job)                                            
                    PJCT[job] = int(time.time() - job_start[job])
           

############### first clear finish status of all jobs ####################

finish_dict = {}
with open('finish.json', 'r') as fp:
    finish_dict = json.load(fp)
for key in finish_dict:
    finish_dict[key] = 0
json_file = json.dumps(finish_dict)
with open('finish.json', 'w') as fp:
    fp.write(json_file)

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

power_dict = {}
with open('power.json', 'r') as fp:
    power_dict = json.load(fp)
for key in power_dict:
    power_dict[key] = 0
json_file = json.dumps(power_dict)
with open('power.json', 'w') as fp:
    fp.write(json_file) 

run_pid_dict = {}
with open('run_pid.json', 'r') as fp:
    run_pid_dict = json.load(fp)
for key in run_pid_dict:
    run_pid_dict[key] = 0
json_file = json.dumps(run_pid_dict)
with open('run_pid.json', 'w') as fp:
    fp.write(json_file) 

ckpt_qual_dict = {}
with open('ckpt_qual.json', 'r') as fp:
    ckpt_qual_dict = json.load(fp)
for key in ckpt_qual_dict:
    ckpt_qual_dict[key] = 0
json_file = json.dumps(ckpt_qual_dict)
with open('ckpt_qual.json', 'w') as fp:
    fp.write(json_file) 

#start_job(K80_node, '0', '39')
#save_job(K80_node, '39')

######################################################################

while True:
    
    # termination condition: 
    # all the jobs have finished

    ################### check for finished jobs on K80 and V100 ##############################

    with open('finish.json', 'r') as fp:
        finish_dict = json.load(fp)

    for gpu, job in K80_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                K80_used -= 1            
                K80_job[gpu] = 'idle'
                print('K80 finished job: ' + job)
                JCT[job] = int(time.time() - job_start[job])  
                # if the job is not qualified for promotion, kill its run.sh processes
                if job not in qualified_job:
                    kill_job(K80_node, job) 

    for gpu, job in V100_job.items():
        if job != 'idle':
            if finish_dict['job'+job] == 1:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                print('V100 finished job: ' + job)
                JCT[job] = int(time.time() - job_start[job])

    ################ check for practical finished jobs on K80 and V100 ######################

    all_job = list(K80_job.values()) + list(V100_job.values())
    check_practical_complete(all_job)

    ################ check run time of current K80 job, update qualified_job #################

    with open('power.json', 'r') as fp:
        power_dict = json.load(fp)   

    for job in list(K80_job.values()):
        if job not in qualified_job and job != 'idle':
            pwr_meas = power_dict['job'+job]
            if pwr_meas > 0:
                qualified_job.append(job)
                print('job' + job + ' has been qualified for promotion')

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    K80_free = K80_cap - K80_used
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
    promote_list = list(set(qualified_job).intersection(list(K80_job.values())).difference(pc_job))
    # this returns job forced to be demoted. Currently in V100, and is practically complete
    force_demote = list(set(list(V100_job.values())).intersection(pc_job))

    if len(promote_list) > 0:
        promoted, demoted = max_power_promotion(K80_free, V100_free, V100_job, promote_list, force_demote)
        if len(promoted) > 0:
            print('promoted jobs: ', promoted)
        if len(demoted) > 0:
            print('demoted jobs: ', demoted)
        # stop all promoted jobs on K80
        for gpu, job in K80_job.items():
            if job in promoted:
                save_job(K80_node, job)
                K80_job[gpu] = 'idle'
                K80_used -= 1
                time.sleep(5)
                
        # stop all demoted jobs on V100
        for gpu, job in V100_job.items():
            if job in demoted:
                save_job(V100_node, job)
                V100_job[gpu] = 'idle'
                V100_used -= 1
                time.sleep(5)

        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in promoted[:]:
            for gpu, job in V100_job.items():
                if job == 'idle': # if gpu idle, schedule new job here
                    resume_job(V100_node, gpu, job_new)
                    V100_job[gpu] = job_new
                    promoted.remove(job_new)
                    V100_used += 1
                    time.sleep(5)
                    break
            
        # resume demoted jobs on K80, make sure the gpu is idle
        for job_new in demoted[:]:
            for gpu, job in K80_job.items():
                if job == 'idle': # if gpu idle, schedule new job here
                    resume_job(K80_node, gpu, job_new)
                    K80_job[gpu] = job_new
                    demoted.remove(job_new)
                    K80_used += 1
                    time.sleep(5)
                    break

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
            if index < len(queue):
                job_new = str(queue[index])
                for gpu, job in K80_job.items():
                    if job == 'idle': # schedule new job here if idle
                        start_job(K80_node, gpu, job_new)
                        measure_job(K80_node, gpu, job_new)
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

average_PJCT = np.average(list(PJCT.values()))
PJCT['average'] = average_PJCT

print('finished all runs')
JCT_name = testcase + '_JCT.json'
PJCT_name = testcase + '_PJCT.json'
with open(JCT_name, 'w') as fp1:
    json.dump(JCT, fp1, sort_keys=True, indent=4)
with open(PJCT_name, 'w') as fp2:
    json.dump(PJCT, fp2, sort_keys=True, indent=4)




