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

queue = [6, 33, 4, 43, 15, 47, 18, 42, 35, 40, 34, 20, 9, 29, 19, 22, 3, 5, 38, 7, 41, 39, 46, 17, 24, 28, 26, 45, 16, 14, 50, 48, 36, 27, 32, 8, 10, 49, 2, 12, 23, 1, 37, 31, 44, 21, 30, 11, 13, 25] 
queue_dict = {}
arrival_time = 0 
for item in queue:
    arrival_time += np.random.poisson(30)
    queue_dict[item] = arrival_time
queue_timer = time.time()

job_start = {} #{'49': time1, '15': time2...}
JCT = {}
PJCT = {} # practical complete time, not applicable for all jobs
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
V100_node = 'd1020'
testcase = args.tc
### also, change .h5 file folder in jobs ###

INTERVAL = 30 # make decision every 30s
QUALIFY_TIME = 300 # 600s or 10min as threshold

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

def max_param_to_idle(V100_free, promote_list):
    param_dict = {}
    with open('param.json', 'r') as fp:
        param_dict = json.load(fp)
    pool_dict = {}
    for job in promote_list:
        if 'job'+job in param_dict:
            pool_dict[job] = param_dict['job'+job]
    # sort the jobs by parameter size 
    sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:V100_free]
    # return the jobs promoted to idle V100s
    return sorted_pool

def max_param_one(K80_free, single_job, promote_list, force_demote):
    param_dict = {}
    with open('param.json', 'r') as fp:
        param_dict = json.load(fp)   
    # first check if job is force demote
    if single_job in force_demote:
        if len(promote_list) > 0: # promotion and demotion
            pool_dict = {}
            for job in promote_list:
                if 'job'+job in param_dict:
                    pool_dict[job] = param_dict['job'+job]
            # sort the jobs by parameter size 
            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:1]         
            promoted = list(set(promote_list).intersection(sorted_pool))
            demoted = list(set([single_job]).difference(sorted_pool))
            return promoted, demoted
        elif K80_free > 0: # demotion only
            return [], [single_job]
        else: # no migration
            return [], []
    else: # compare parameter
        if len(promote_list) > 0:
            V100_pool = list(set([single_job]).union(promote_list))
            pool_dict = {}
            for job in V100_pool:
                if 'job'+job in param_dict:
                    pool_dict[job] = param_dict['job'+job]        
            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:1]
            promoted = list(set(promote_list).intersection(sorted_pool))
            demoted = list(set([single_job]).difference(sorted_pool))
            return promoted, demoted
        else:
            return [], []


#def max_param_promotion(K80_free, V100_free, V100_job, promote_list, force_demote):
#    num_demote = len(force_demote)
#    num_promote = len(promote_list)  
#    V100_vacant = num_demote + V100_free
#    K80_vacant = num_promote + K80_free 
#    param_dict = {}
#    with open('param.json', 'r') as fp:
#        param_dict = json.load(fp)
#    if K80_vacant >= num_demote: # if more vacant K80s than demote jobs, always demote
#        # selectively promote among active V100 jobs and promote list jobs
#        V100_qual = list(set(list(V100_job.values())) - set(force_demote))
#        if 'idle' in V100_qual:
#            V100_qual.remove('idle')
#        V100_pool = list(set(V100_qual).union(promote_list))       
#        if len(V100_pool) <= 4: # promote all jobs as well
#            return promote_list, force_demote
#        else: # promote the top 4 jobs            
#            pool_dict = {}
#            for job in V100_pool:
#                if 'job'+job in param_dict:
#                    pool_dict[job] = param_dict['job'+job]        
#            sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:4] 
#            promotion_list = list(set(promote_list).intersection(sorted_pool))                     
#            demotion_list = list(set(list(V100_job.values())).difference(sorted_pool))
#            if 'idle' in demotion_list:
#                demotion_list.remove('idle') # this includes force demotion
#            return promotion_list, demotion_list
#    elif V100_vacant >= num_promote: # if more vacant V100s than promote jobs, always promote
#        # less vacant K80s than demote jobs, select worst among force demote list
#        pool_dict = {} # here the pool only includes force demote jobs
#        for job in force_demote:
#             if 'job'+job in param_dict:
#                pool_dict[job] = param_dict['job'+job]
#        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=False)[:K80_vacant] 
#        return promote_list, sorted_pool
#    else:
#        raise ValueError('Bug with max param promotion, condition not considered')

#a, b = max_param_promotion(1, 1, {0: '49', 1: '39', 2: '50', 3: 'idle'}, ['40', '37'], []) 
#c, d = max_param_promotion(1, 1, {0: 'idle', 1: 'idle', 2: 'idle', 3: 'idle'}, [], [])
#e, f = max_param_promotion(1, 1, {'0': '49', '1': '39', '2': '50', '3': 'idle'}, ['40', '37'], []) 

def save_job(node, job): # save_job('c2176', '50')
    # first wait for the job to be qualified for checkpointing
    while True: # wait for ckpt_qual.json to be available
        if os.path.exists('ckpt_qual.json'):
            with open('ckpt_qual.json', 'r') as fp2:
                ckpt_qual_dict = json.load(fp2)
            if ckpt_qual_dict['job'+job] == 1:
                if os.path.exists('ckpt_qual.json'): #if not locked. lock it and edit
                    try:
                        os.rename('ckpt_qual.json', 'ckpt_qual_lock.json') # lock
                        with open('ckpt_qual_lock.json', 'r') as fp2:
                            ckpt_qual_lock_dict = json.load(fp2)
                        ckpt_qual_lock_dict['job'+job] = 0 # reset it
                        json_file = json.dumps(ckpt_qual_lock_dict)
                        with open('ckpt_qual_lock.json', 'w') as fp2:
                            fp2.write(json_file)
                        os.rename('ckpt_qual_lock.json', 'ckpt_qual.json')
                        break
                    except Exception:
                        pass
        time.sleep(5)

    send_signal(node, 'save ' + job)

    global ovhd_start
    global pc_job
    #if job not in pc_job:
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
        with open('finish.json', 'r') as fp3:
            finish_dict = json.load(fp3)
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

    while True:
        if os.path.exists('param.json'):
            os.rename('param.json', 'param_lock.json')
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

    while True:
        if os.path.exists('param.json'):
            break
        else:
            time.sleep(1)
   
# function that checks the tensorboard log of currently running jobs and logs practical complete jobs in a global list
# once a job reaches practical complete, it cannot be promoted. If it's already promoted, it gets demoted.
# criteria for practical complete: loss improvement has been smaller than 0.01 for last 3 consecutive epochs
def check_practical_complete(job_list):
    log_path = '/scratch/li.baol/tsrbrd_log/job_runs/' + testcase + '/'
    threshold = 0.01
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

param_dict = {}
with open('param.json', 'r') as fp:
    param_dict = json.load(fp)
for key in param_dict:
    param_dict[key] = 0
json_file = json.dumps(param_dict)
with open('param.json', 'w') as fp:
    fp.write(json_file) 

ckpt_qual_dict = {}
with open('ckpt_qual.json', 'r') as fp:
    ckpt_qual_dict = json.load(fp)
for key in ckpt_qual_dict:
    ckpt_qual_dict[key] = 0
json_file = json.dumps(ckpt_qual_dict)
with open('ckpt_qual.json', 'w') as fp:
    fp.write(json_file) 

#files = glob.glob('epoch/job*.txt') 
#files.sort(key=lambda x: os.path.getmtime(x))
#pdb.set_trace()
#job = files[0].split('/')[1].split('.')[0].replace('job', '')

# remove all files in epoch folder
files = glob.glob('epoch/*')
for f in files:
    os.remove(f)

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
            elif ovhd_start[job] != 0:
                # check if ckpt overhead has finished
                if os.path.exists('ckpt_qual.json'):
                    with open('ckpt_qual.json', 'r') as fp2:
                        ckpt_qual_dict = json.load(fp2)
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
                if os.path.exists('ckpt_qual.json'):
                    with open('ckpt_qual.json', 'r') as fp2:
                        ckpt_qual_dict = json.load(fp2)
                    if ckpt_qual_dict['job'+job] == 1:
                        overhead[job] += int(time.time() - ovhd_start[job])
                        ovhd_start[job] = 0

    ################ check for practical finished jobs on K80 and V100 ######################

    all_job = list(K80_job.values()) + list(V100_job.values())
    check_practical_complete(all_job)

    ################ check run time of current K80 job, update qualified_job #################

    for job in list(K80_job.values()):
        if job not in qualified_job and job != 'idle':
            runtime = int(time.time() - job_start[job])
            if runtime >= QUALIFY_TIME:
                qualified_job.append(job)
                print('job' + job + ' has been qualified for promotion')

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
    promote_list = list(set(qualified_job).intersection(list(K80_job.values())).difference(pc_job))
    # this returns job forced to be demoted. Currently in V100, and is practically complete
    force_demote = list(set(list(V100_job.values())).intersection(pc_job))

    # first for empty gpus, promote from promote_list
    promote_idle = max_param_to_idle(V100_free, promote_list)
    promote_list_new = list(set(promote_list).difference(promote_idle))

    if len(promote_idle) > 0:
        print('promoted jobs to idle: ', promote_idle)
        # stop all promoted jobs on K80
        for gpu, job in K80_job.items():
            if job in promote_idle:
                save_job(K80_node, job)
                K80_job[gpu] = 'idle'
                K80_used -= 1

        # resume promoted jobs on V100, make sure the gpu is idle
        for job_new in promote_idle[:]:
            for gpu, job in V100_job.items():
                if job == 'idle': # if gpu idle, schedule new job here
                    resume_job(V100_node, gpu, job_new)
                    #if job_new not in pc_job:
                    num_mig[job_new] += 1
                    V100_job[gpu] = job_new
                    promote_idle.remove(job_new)
                    V100_used += 1
                    break

    K80_free = K80_cap - K80_used

    # check for jobs with epoch end

    files = glob.glob('epoch/job*.txt') 
    files.sort(key=lambda x: os.path.getmtime(x))

    V100_epoch = []

    if len(files) > 0:
        job = files[0].split('/')[1].split('.')[0].replace('job', '')
        os.remove(files[0])
        if job in V100_job.values() and job not in promote_idle:
            V100_epoch.append(job)
        if len(V100_epoch) > 0:
            promoted_job, demoted_job = max_param_one(K80_free, V100_epoch[0], promote_list_new, force_demote)

            if len(promoted_job) > 0:
                print('promoted jobs: ', promoted_job)
                for gpu, job in K80_job.items():
                    if job in promoted_job:
                        save_job(K80_node, job)
                        K80_job[gpu] = 'idle'
                        K80_used -= 1          
            if len(demoted_job) > 0:
                print('demoted jobs: ', demoted_job)
                for gpu, job in V100_job.items():
                    if job in demoted_job:
                        save_job(V100_node, job)
                        V100_job[gpu] = 'idle'
                        V100_used -= 1          
            if len(promoted_job) > 0:
                job_new = promoted_job[0]
                for gpu, job in V100_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        resume_job(V100_node, gpu, job_new)
                        #if job_new not in pc_job:
                        num_mig[job_new] += 1
                        V100_job[gpu] = job_new
                        promoted_job.remove(job_new)
                        V100_used += 1
                        break
            if len(demoted_job) > 0:
                job_new = demoted_job[0]
                for gpu, job in K80_job.items():
                    if job == 'idle': # if gpu idle, schedule new job here
                        resume_job(K80_node, gpu, job_new)
                        #if job_new not in pc_job:
                        num_mig[job_new] += 1
                        K80_job[gpu] = job_new
                        demoted_job.remove(job_new)
                        K80_used += 1
                        break
            # perform a check, make sure all promoted/demoted jobs are scheduled
            if len(promoted_job) > 0 or len(demoted_job) > 0:
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

    time.sleep(1)

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

average_overhead = np.average(list(overhead.values()))
overhead['average'] = average_overhead

print('finished all runs')
JCT_name = testcase + '_JCT.json'
PJCT_name = testcase + '_PJCT.json'
overhead_name = testcase + '_overhead.json'
num_mig_name = testcase + '_num_mig.json'
with open(JCT_name, 'w') as fp1:
    json.dump(JCT, fp1, sort_keys=True, indent=4)
with open(PJCT_name, 'w') as fp2:
    json.dump(PJCT, fp2, sort_keys=True, indent=4)
with open(overhead_name, 'w') as fp3:
    json.dump(overhead, fp3, sort_keys=True, indent=4)
with open(num_mig_name, 'w') as fp3:
    json.dump(num_mig, fp3, sort_keys=True, indent=4)

