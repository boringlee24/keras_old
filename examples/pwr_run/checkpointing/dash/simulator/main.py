import pdb
import time
import os
import subprocess
import re
import random
import json
import numpy as np
import glob
import socket
import argparse
import threading
import _thread
import signal
from datetime import datetime
import csv
import opt_dash

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument("--ovhd", action="store_true", default=False)
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='select testcase')
args = parser.parse_args()
testcase = args.tc

# this is a simulator
# simulation options: 1. highest speedup promotion 2. most time save promotion 3. shortest job first promotion

STEP_SIZE = 30
Tnow = 0 # current passed time
run_log = open('run.log','w')

with open('../job_trace/job_queue_sc_50.json', 'r') as fp: #TODO
    queue = json.load(fp)
queue_dict = {} # contains job arrive time
arrival_time = 0 
for item in queue:
    arrival_time += np.random.poisson(30)
    queue_dict[item] = arrival_time
queue_timer = Tnow
queue_delay = {}
for item in queue:
    queue_delay[str(item)] = 0

# need these information
# JRT of each job on K80 and V100 (available)
# mig overhead of each job from K80-V100 and V100-K80 (need to collect)
# 1st batch extra time for each job on K80 and V100 (need to collect)
# ignore epoch wasted time here

with open('job_info/k80_only_JCT.json', 'r') as fp: 
    k80_jrt = json.load(fp)
with open('job_info/v100_only_JCT.json', 'r') as fp: 
    v100_jrt = json.load(fp)

#K80_JRT = {}
#for item in queue:
#    K80_JRT[str(item)] = 0
#V100_JRT = {}
#for item in queue:
#    V100_JRT[str(item)] = 0

if args.ovhd:
    print('error, no overhead collected')
    exit()
else:
    mig_ovhd = {}
    for item in queue:
        mig_ovhd[str(item)] = 0
    K80_1st_batch = {}
    for item in queue:
        K80_1st_batch[str(item)] = 0
    V100_1st_batch = {}
    for item in queue:
        V100_1st_batch[str(item)] = 0

# no support for 2-gpu jobs
multigpu_list = []#['1', '2', '3']#, '4', '5', '6', '7'] #TODO

job_start = {} #{'49': time1, '15': time2...} # start time of job
JRT = {}
for item in queue:
    JRT[str(item)] = 0
completion = {}
for item in queue:
    completion[str(item)] = 0
num_mig = {} # initialize migration time to 0
for item in queue:
    num_mig[str(item)] = 0
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

speedup_dict = {} # K80_JRT / V100_JRT
for item in queue:
    speedup_dict[str(item)] = k80_jrt[str(item)] / v100_jrt[str(item)]
K80_remain = {} # job remain time on K80, set to jrt when job starts
for item in queue:
    K80_remain[str(item)] = 0 
V100_remain = {}
for item in queue:
    V100_remain[str(item)] = 0 

birthplace = {}
for item in queue:
    birthplace[str(item)] = 'none' 

index = 0

K80_cap = 4 #TODO
V100_cap = 8
K80_used = 0
V100_used = 0
K80_per_node = 8
V100_per_node = 4
K80_node = ['K1']#, 'c2182']
V100_node = ['V1', 'V2']#, 'd1015']

K80_job = {}
for i in range(K80_cap):
    K80_job[str(i)] = 'idle'
V100_job = {}
for i in range(V100_cap):
    V100_job[str(i)] = 'idle'

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

def get_remaining_time(job_list, K80_remain, V100_remain):
    return_dict = {}
    for job in job_list:
        return_dict[job] = [K80_remain[job], V100_remain[job]]
    return return_dict

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
    group_2gpu = [i for i in res_group if len(i) == 2] # 2gpu id [['1','2'],['4','7']]
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

# checks if 2-GPU jobs can be promoted/demoted without locality issue
# if cannot, remove 2-GPU job and corresponding 1-GPU job until all jobs can fit
# then returns new_K80_avail, new_V100_avail, new_promoted, new_demoted
def locality_check(K80_avail, V100_avail, promoted, demoted):
    '''
    K80/V100_avail: ['1', '2', '5']
    promoted/demoted: ['7','7','50','70'] 
    '''
    for item in range(2):#[K80_avail, V100_avail]:
        skip = False
        res_group = [] # group reserved GPU together
        GPU_avail = [K80_avail,V100_avail][item]
        for i in range(len(GPU_avail)):
            if skip:
                skip = False
                continue
            else:
                # two gpus from the same node
                if item == 0:
                    GPU_per_node = K80_per_node
                elif item == 1:
                    GPU_per_node = V100_per_node
                if i!=len(GPU_avail)-1 and int(GPU_avail[i])//GPU_per_node==int(GPU_avail[i+1])//GPU_per_node:
                    skip = True
                    res_group.append([GPU_avail[i], GPU_avail[i+1]])
                else:
                    res_group.append([GPU_avail[i]])
        if item == 0:
            K80_1gpu = [i for i in res_group if len(i) == 1] # 1gpu id
            K80_2gpu = [i for i in res_group if len(i) == 2] # 2gpu id [['1','2'],['4','7']]
        elif item == 1:
            V100_1gpu = [i for i in res_group if len(i) == 1] # 1gpu id
            V100_2gpu = [i for i in res_group if len(i) == 2] # 2gpu id

    promoted_1gpu = [i for i in promoted if i not in multigpu_list] # 1gpu job
    promoted_2gpu = [i for i in promoted if i in multigpu_list] # 2gpu job ['3','3','4','4','10']
    demoted_1gpu = [i for i in demoted if i not in multigpu_list] # 1gpu job
    demoted_2gpu = [i for i in demoted if i in multigpu_list] # 2gpu job

    condition1 = len(K80_avail) >= len(demoted) and 2*len(K80_2gpu) >= len(demoted_2gpu)
    condition2 = len(V100_avail) >= len(promoted) and 2*len(V100_2gpu) >= len(promoted_2gpu)

    if condition1 and condition2:
        return None
    else:
        print('Notice: promoted/demoted jobs cannot fit in their destination due to locality', file=run_log, flush=True)
        print('Remove all 2-gpu jobs from this migration decision', file=run_log, flush=True) # meaning they stay wherever they were before
        for job in promoted_2gpu:
            promoted.remove(job)
        for job in demoted_2gpu:
            demoted.remove(job)
        for gpu_pair in K80_2gpu:
            for gpu in gpu_pair:
                K80_avail.remove(gpu)
        for gpu_pair in V100_2gpu:
            for gpu in gpu_pair:
                V100_avail.remove(gpu)
        # check if need to remove 1-gpu jobs as well
        if len(K80_avail) < len(demoted_1gpu):
            diff = len(demoted_1gpu) - len(K80_avail)
            for i in range(diff):
                removed_1gpu = demoted[0]
                demoted.remove(removed_1gpu)
                # also need to remove its corresponding GPU
                V100_avail.remove(demoted_V100_map_1gpu[removed_1gpu])
        elif len(V100_avail) < len(promoted_1gpu):
            diff = len(promoted_1gpu) - len(V100_avail)
            for i in range(diff):
                removed_1gpu = promoted[0]
                promoted.remove(removed_1gpu)
                # also need to remove its corresponding GPU
                K80_avail.remove(promoted_K80_map_1gpu[removed_1gpu])

        return K80_avail, V100_avail, promoted, demoted

while True:
    # termination condition: 
    # all the jobs have finished
    ################### check for finished jobs on K80 and V100 ##############################

    for gpu, job in K80_job.items():
        if job != 'idle':
            if K80_remain[job] <= 0:
                K80_used -= 1            
                K80_job[gpu] = 'idle'
                JRT[job] = Tnow - job_start[job]
                K80_time[job] += Tnow - K80_start_time[job]
                completion[job] = 1
                print('K80 finished job: ' + job, file=run_log, flush=True)

    for gpu, job in V100_job.items():
        if job != 'idle':
            if V100_remain[job] <= 0:
                V100_used -= 1            
                V100_job[gpu] = 'idle'
                JRT[job] = Tnow - job_start[job]
                V100_time[job] += Tnow - V100_start_time[job]
                completion[job] = 1
                print('V100 finished job: ' + job, file=run_log, flush=True)

    ################ submit new jobs to vacant GPUs ############################

    # first fill in vacant V100s
    if V100_used < V100_cap:
        V100_free = V100_cap - V100_used
        for i in range(V100_free):
            time_passed = Tnow
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                for gpu, job in V100_job.items():
                    if job == 'idle': # schedule new job here if idle
                        real_node, real_gpu = V100_LUT(gpu)
                        V100_remain[job_new] = v100_jrt[job_new]
                        K80_remain[job_new] = k80_jrt[job_new]
                        birthplace[job_new] = real_node
                        V100_job[gpu] = job_new
                        job_start[job_new] = Tnow
                        queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                        V100_start_time[job_new] = Tnow
                        index += 1
                        V100_used += 1
                        break
    # first fill in vacant K80s
    if K80_used < K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            time_passed = Tnow
            if index < len(queue) and queue_dict[queue[index]] < time_passed: # make sure job has arrived in the queue
                job_new = str(queue[index])
                for gpu, job in K80_job.items():
                    if job == 'idle': # schedule new job here if idle
                        real_node, real_gpu = K80_LUT(gpu)
                        V100_remain[job_new] = v100_jrt[job_new]
                        K80_remain[job_new] = k80_jrt[job_new]
                        birthplace[job_new] = real_node
                        K80_job[gpu] = job_new
                        job_start[job_new] = Tnow
                        queue_delay[job_new] = int(time_passed - queue_dict[queue[index]])
                        K80_start_time[job_new] = Tnow
                        index += 1
                        K80_used += 1
                        break

    ################## make promotion decisions ################
    # figure out which job enters the pool and which GPUs enter the pool
    # job must be in step1_job, and if it's on V100, it must have passed demote_qualify_time
    # the selected job's current GPU also enters GPU pool. And if GPU is idle, it gets added into the pool as well

    job_pool = []
    K80_pool = []
    V100_pool = []
    for gpu, job in K80_job.items():
        if job != 'idle':
            job_pool.append(job)
            K80_pool.append(gpu)
        else:
            K80_pool.append(gpu)
    for gpu, job in V100_job.items():
        if job != 'idle':
            job_pool.append(job)
            V100_pool.append(gpu)
        else:
            V100_pool.append(gpu)
    
    # prepare inputs and perform optimization
    num_GPUs = [len(K80_pool), len(V100_pool)]
    job_num_GPUs = {}
    for job in job_pool:
        job_num_GPUs[job] = 1
    job_remaining_time = get_remaining_time(job_pool, K80_remain, V100_remain)

    promoted = [] # jobs to be placed in V100. 2-gpu jobs are duplicated
    demoted = [] # jobs to be placed in K80

    # perform 1st optimization
    if len(job_num_GPUs) > 0 and len(job_remaining_time) > 0:
        if 'optimize' in testcase:
            opt_decision = opt_dash.optimize_promotion(num_GPUs, job_num_GPUs, job_remaining_time)
        elif 'max_speedup' in testcase:
            opt_decision = opt_dash.max_speedup_first(num_GPUs, job_num_GPUs, job_remaining_time)
        elif 'shortest_job' in testcase:
            opt_decision = opt_dash.shortest_jobs_first(num_GPUs, job_num_GPUs, job_remaining_time)
        else:
            print('testcase does not exist')
            sys.exit()

        print('job_pool:',job_pool,'K80_pool:',K80_pool,'V100_pool:',V100_pool,'remain_time',job_remaining_time,'decision:',opt_decision, file=run_log, flush=True)

        # check if placement of promo/demo 2-gpu jobs are viable
        # if not viable: remove jobs that benefit least from promo/hurt least from demo
        for job in opt_decision:
            placement = opt_decision[job]
            if placement == 1 and job in list(K80_job.values()):
                promoted.append(job)
            elif placement == 0 and job in list(V100_job.values()):
                demoted.append(job)

        if len(promoted) > 0:
            print('promotion', promoted, file=run_log, flush=True)
        if len(demoted) > 0:
            print('demotion', demoted, file=run_log, flush=True)
        if len(demoted) > 0 or len(promoted) > 0:
            # generate K80/V100 GPU list that are either idle or have job in promoted/demoted
            # to be used by placement function
            K80_avail = [] 
            V100_avail = []
            promoted_K80_map_1gpu = {} # original mapping for 1-gpu jobs, used in "locality check"
            demoted_V100_map_1gpu = {}
            for gpu, job in K80_job.items():
                if job == 'idle':
                    K80_avail.append(gpu)
                elif job in promoted:
                    K80_avail.append(gpu)
                    promoted_K80_map_1gpu[job] = gpu
            for gpu, job in V100_job.items():
                if job == 'idle':
                    V100_avail.append(gpu)
                elif job in demoted:
                    V100_avail.append(gpu)
                    demoted_V100_map_1gpu[job] = gpu

            # use these information: K80_avail, V100_avail, promoted, demoted
            check_result = locality_check(K80_avail, V100_avail, promoted, demoted)
            if check_result is not None:
                K80_avail, V100_avail, promoted, demoted = check_result

            # now place promoted jobs on V100_avail and demoted jobs on K80_avail
            K80_mapping = GPU_placement(K80_avail, demoted, gpu_type='K80')
            V100_mapping = GPU_placement(V100_avail, promoted, gpu_type='V100')

    # make promotion decisions
    if len(promoted) > 0 or len(demoted) > 0:
        # remove duplicated 2-gpu jobs from promoted and demoted
        promoted = list(dict.fromkeys(promoted))
        demoted = list(dict.fromkeys(demoted))

        # stop all promoted jobs on K80
        for job in promoted[:]:
            # need to find its current gpu on K80
            current_gpu = ''
            for gpu, job_K in K80_job.items():
                if job_K == job:
                    current_gpu = gpu
                    break
            K80_job[current_gpu] = 'idle'
            K80_used -= 1
            #save_job(real_node, job)
            K80_time[job] += int(Tnow - K80_start_time[job])

        # stop all demoted jobs on V100
        for job in demoted[:]:
            # need to find its current gpu on V100
            current_gpu = ''
            for gpu, job_K in V100_job.items():
                if job_K == job:
                    current_gpu = gpu
                    break
            V100_job[current_gpu] = 'idle'
            V100_used -= 1
            #save_job(real_node, job)
            V100_time[job] += int(Tnow - V100_start_time[job])

        # resume promoted jobs on V100
        for job in promoted[:]:
            gpu = V100_mapping[job]
#            resume_job(real_node, real_gpu, job)
            V100_start_time[job] = Tnow
            V100_job[gpu] = job
            V100_used += 1
            promoted.remove(job)
            num_mig[job] += 1

        # resume demoted jobs on K80
        for job in demoted[:]:
            gpu = K80_mapping[job]
#            resume_job(real_node, real_gpu, job)
            K80_start_time[job] = Tnow
            K80_job[gpu] = job
            K80_used += 1
            demoted.remove(job)
            num_mig[job] += 1

        # perform a check, make sure all promoted/demoted jobs are scheduled
        if len(promoted) > 0 or len(demoted) > 0:
            raise ValueError('Bug with promotion scheme, more jobs than free gpus')

    ############## monitor GPU usage ############

    usage = K80_used + V100_used
    time_stamp = int(Tnow - queue_timer)
    gpu_usage_time.append(time_stamp)
    gpu_usage.append(usage)
    total_completion = np.sum(list(completion.values()))
    gpu_usage_completion.append(total_completion)

    ############### wait for next iteration, job is running ##########

    Tnow += STEP_SIZE
    # for each job currently running, reduce its remaining time proportionally
    for gpu, job in K80_job.items():
        if job != 'idle':
            K80_remain[job] -= STEP_SIZE
            V100_remain[job] = K80_remain[job] / speedup_dict[job]
            completion[job] = round(1 - K80_remain[job] / k80_jrt[job], 2)
    for gpu, job in V100_job.items():
        if job != 'idle':
            V100_remain[job] -= STEP_SIZE
            K80_remain[job] = V100_remain[job] * speedup_dict[job]
            completion[job] = round(1- V100_remain[job] / v100_jrt[job], 2)
    
    ################ check if termination condition is met ################

    K80_idle_num = sum(value == 'idle' for value in K80_job.values())
    V100_idle_num = sum(value == 'idle' for value in V100_job.values())
    if K80_idle_num == K80_cap and V100_idle_num == V100_cap and index == len(queue):
        print('all jobs are finished!', file=run_log, flush=True)
        break

###########################

# get average JRT
average_JRT = np.average(list(JRT.values()))
JRT['average'] = average_JRT

average_queue_delay = np.average(list(queue_delay.values()))
queue_delay['average'] = average_queue_delay

# after everything is finished

print('finished all runs', file=run_log, flush=True)
JRT_name = testcase + '_JRT.json'
num_mig_name = testcase + '_num_mig.json'
K80_time_name = testcase + '_K80_time.json'
V100_time_name = testcase + '_V100_time.json'
gpu_usage_name = testcase + '_gpu_usage.csv'
completion_name = 'completion.json'
queue_delay_name = testcase + '_queue_delay.json'
birthplace_name = testcase + '_birthplace.json'
speedup_name = testcase + '_speedup.json'
K80_remain_name = testcase + '_K80_remain.json'
V100_remain_name = testcase + '_V100_remain.json'

with open(JRT_name, 'w') as fp1:
    json.dump(JRT, fp1, sort_keys=True, indent=4)
with open(num_mig_name, 'w') as fp3:
    json.dump(num_mig, fp3, sort_keys=True, indent=4)
with open(K80_time_name, 'w') as fp3:
    json.dump(K80_time, fp3, sort_keys=True, indent=4)
with open(V100_time_name, 'w') as fp3:
    json.dump(V100_time, fp3, sort_keys=True, indent=4)
with open(completion_name, 'w') as fp1:
   json.dump(completion, fp1, sort_keys=True, indent=4)
with open(queue_delay_name, 'w') as fp1:
   json.dump(queue_delay, fp1, sort_keys=True, indent=4)
with open(birthplace_name, 'w') as fp1:
   json.dump(birthplace, fp1, sort_keys=True, indent=4)
with open(speedup_name, 'w') as fp1:
   json.dump(speedup_dict, fp1, sort_keys=True, indent=4)
with open(K80_remain_name, 'w') as fp3:
    json.dump(K80_remain, fp3, sort_keys=True, indent=4)
with open(V100_remain_name, 'w') as fp3:
    json.dump(V100_remain, fp3, sort_keys=True, indent=4)

gpu_usage_time = np.asarray(gpu_usage_time)
gpu_usage = np.asarray(gpu_usage)
gpu_usage_completion = np.asarray(gpu_usage_completion)
rows = zip(gpu_usage_time, gpu_usage, gpu_usage_completion)
with open(gpu_usage_name, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

