import pdb
import time
import os
import subprocess
import re
import random
import json
import numpy as np

queue = [49, 15, 50, 39, 14, 40, 13, 37, 32, 44, 1, 25, 6, 12, 43, 35, 29, 7, 46, 23, 47, 34, 21, 33, 36, 24, 28, 48, 17, 8, 45, 30, 2, 41, 16, 3, 27, 20, 38, 11, 42, 10, 22, 4, 18, 19, 5, 9, 26, 31]
job_start = {} #{'49': time1, '15': time2...}
JCT = {} 
index = 0

K80_cap = 8
V100_cap = 4
K80_used = 0
V100_used = 0
qualified_jobs = 0

K80_job = []
V100_job = []
qualified_job = []

testcase = 'random_no_pratical_finish'

INTERVAL = 30 # make decision every 30s
QUALIFY_TIME = 600 # 600s or 10min as threshold

# takes in a list of jobs qualified for promote, returns a list of jobs that get upgraded, and an empty list for demoted
# jobs
def random_promotion(V100_free, promote_list):
    if V100_free >= len(promote_list):
        return promote_list, []
    else:
        return random.sample(promote_list, V100_free), []

# checks squeue every 10s to see if the job has ended
def wait_till_job_ends(job):
    CHECK_INTERVAL = 10 # 10s
    while True:
        stdout = subprocess.check_output(['squeue -u $USER | awk \"/job/\" | awk \'{print $3}\''], shell=True)
        stdout = str(stdout)
        job_run = re.findall(r'\d+', stdout) # this is list of currently running jobs in string, e.g. ['20', '48'...]
        if job not in job_run:
            break
        time.sleep(CHECK_INTERVAL)


while True:
    
    # termination condition: 
    # all the jobs have finished

    ################### check for finished jobs on K80 and V100 ##############################

    # check list of currently running K80 jobs
    stdout = subprocess.check_output(['squeue -u $USER | awk \"/_K/\" | awk \'{print $3}\''], shell=True)
    stdout = str(stdout)
    K80_run = re.findall(r'\d+', stdout) # this is list of currently running jobs in string
    # now confirm if there are jobs that are finished
    # this syncs K80_job and K80_run
    for job in K80_job[:]:
        if job not in K80_run:
            K80_used -= 1            
            K80_job.remove(job) 
            JCT[job] = int(time.time() - job_start[job])
    
    # check list of currently running V100 jobs
    stdout = subprocess.check_output(['squeue -u $USER | awk \"/_V/\" | awk \'{print $3}\''], shell=True)
    stdout = str(stdout)
    V100_run = re.findall(r'\d+', stdout) # this is list of currently running jobs in string
    # now confirm if there are jobs that are finished
    # this syncs V100_job and V100_run
    for job in V100_job[:]:
        if job not in V100_run:
            V100_used -= 1            
            V100_job.remove(job)
            JCT[job] = int(time.time() - job_start[job])

    ################ check run time of current running new K80 jobs #################

    for job in K80_job:
        if job not in qualified_job:
            runtime = int(time.time() - job_start[job])
            if runtime >= QUALIFY_TIME:
                qualified_job.append(job)

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    # this returns available jobs for promotion
    promote_list = list(set(qualified_job).intersection(K80_job))

    if len(promote_list) > 0:
        promoted, demoted = random_promotion(V100_free, promote_list)
        # stop all promoted jobs on K80
        for job in promoted:
            cmd = 'scancel --signal=TERM --name=job' + job + '_K' # scancel --signal=TERM --name=job1_4gpu
            subprocess.call([cmd], shell=True)
            K80_job.remove(job)
            K80_used -= 1
        # stop all demoted jobs on V100
        for job in demoted:
            cmd = 'scancel --signal=TERM --name=job' + job + '_V'
            subprocess.call([cmd], shell=True)
            V100_job.remove(job)
            V100_used -= 1

        # resume promoted jobs on V100
        for job in promoted:
            cmd = './run_resume.sh job' + job + ' V ' + testcase
            wait_till_job_ends(job)
            subprocess.call([cmd], shell=True)
            V100_job.append(job)
            V100_used += 1
        # resume demoted jobs on K80
        for job in demoted:
            cmd = './run_resume.sh job' + job + ' K ' + testcase
            wait_till_job_ends(job)
            subprocess.call([cmd], shell=True)
            K80_job.append(job)
            K80_used += 1

    ################ submit new jobs to vacant K80 GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
    if K80_used != K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            if index < len(queue):
                job = str(queue[index])
                cmd = './run_start.sh job' + job + ' K ' + testcase
#                os.system(cmd)
                subprocess.call([cmd], shell=True)
                K80_job.append(job)
                job_start[job] = time.time()
                index += 1
                K80_used += 1


    ################ check if termination condition is met ################

    if len(K80_job) == 0 and len(V100_job) == 0 and index == len(queue):
        break

    # this loop performs actions every 30s
    time.sleep(INTERVAL)

# get average JCT
average_JCT = np.average(list(JCT.values()))
JCT['average'] = average_JCT

print('finished all runs')
filename = testcase + '.json'
with open(filename, 'w') as fp:
    json.dump(JCT, fp, sort_keys=True, indent=4)
