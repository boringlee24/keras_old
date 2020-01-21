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

K80_job = []
V100_job = []
all_job = []
qualified_job = []
pc_job = [] # list of jobs that are pratically completed

nodelist='c2178,d1018' ### change this ###
testcase = 'max_pwr_0.001' ### change this ###
### also, change .h5 file folder in jobs ###

INTERVAL = 60 # make decision every 30s
QUALIFY_TIME = 600 #600s or 10min as threshold

# takes in a list of jobs qualified for promote, returns a list of jobs that get upgraded, and a list for demoted jobs
def random_promotion(V100_free, promote_list, force_demote):
    num_demote = len(force_demote)
    V100_avail = num_demote + V100_free
    if V100_avail >= len(promote_list):
        return promote_list, force_demote
    else:
        return random.sample(promote_list, V100_avail), force_demote

def max_power_promotion(V100_job, promote_list, force_demote):
    power_dict = {}
    with open('power.json', 'r') as fp:
        power_dict = json.load(fp)
    V100_qual = list(set(V100_job) - set(force_demote))
    V100_pool = V100_qual + promote_list # any current V100 job not demoted and K80 job qualified can go into the pool
    num_V100_pool = len(V100_pool)
    if num_V100_pool <= 4:
        # every thing in the pool should go into V100
        return promote_list, force_demote
    else:
        pool_dict = {}
        for job in V100_pool:
            # make sure the key exists 
            if 'job'+job in power_dict:
                pool_dict[job] = power_dict['job'+job] 
        # sort dict keys from big value to small value, and take the first 4 only
        sorted_pool = sorted(pool_dict, key=pool_dict.get, reverse=True)[:4] 
        promotion_list = list(set(promote_list).intersection(sorted_pool))
        demotion_list = list(set(V100_job).difference(sorted_pool))
        return promotion_list, demotion_list

# checks squeue every 10s to see if the job has ended
def wait_till_job_ends(job, gpu):
    CHECK_INTERVAL = 10 # 10s
    cmd = 'squeue -u $USER --name=job' + job + '_' + gpu + ' --nodelist=' + nodelist + ' | awk \"/job/\" | awk \'{print $3}\''
    while True:
        try:
            stdout = subprocess.check_output([cmd], shell=True)        
            stdout = str(stdout)
            job_run = re.findall(r'\d+', stdout) # this is list of currently running jobs in string, e.g. ['20', '48'...]
            if job not in job_run:
                break
        except: #subprocess.CalledProcessError:
            print('encountered squeue error when waiting for job to end')
        time.sleep(CHECK_INTERVAL)
        scancel_job(job, gpu)
        time.sleep(CHECK_INTERVAL)
        

# checks output and makes sure no exception occurs
def check_output(cmd):
    RETRY_INTERVAL = 10
    while True:
        try:
            stdout = subprocess.check_output([cmd], shell=True)
            return stdout
        except: #subprocess.CalledProcessError:
            print('encountered scancel error when calling check_output function, retrying...')
            time.sleep(RETRY_INTERVAL)          

# cancel job, and make sure there is no error with scancel command
def scancel_job(job, gpu): # scancel_job('50', 'K')
    RETRY_INTERVAL = 10
    cmd = 'scancel --signal=TERM --name=job' + job + '_' + gpu + ' --nodelist=' + nodelist
    while True:
        try:
            subprocess.check_output([cmd], shell=True)
            break
        except: #subprocess.CalledProcessError:
            print('encountered scancel error, retrying...')
            time.sleep(RETRY_INTERVAL)

# kill job, and make sure there is no error with scancel command
def kill_job(job, gpu): # scancel_job('50', 'K')
    RETRY_INTERVAL = 10
    cmd = 'scancel --name=job' + job + '_' + gpu + ' --nodelist=' + nodelist
    while True:
        try:
            subprocess.check_output([cmd], shell=True)
            break
        except: #subprocess.CalledProcessError:
            print('encountered scancel error, retrying...')
            time.sleep(RETRY_INTERVAL)

# resume job, and make sure there is no error with sbatch submission
def resume_job(job, gpu): # resume_job('1', 'V')
    RETRY_INTERVAL = 10
    cmd = './run_resume.sh job' + job + ' ' + gpu + ' ' + testcase
    while True:
        stdout = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE).stdout
        stdout = str(stdout)      
        print(stdout)
        if 'Submitted batch job' in stdout:
            break
        else:
            kill_job(job, gpu) # make sure repeated submission doesn't exist
            print('encountered sbatch error on job resume, retrying...')
        time.sleep(RETRY_INTERVAL)

# start job, and make sure there is no error with sbatch submission
def start_job(job): # start_job('1')
    RETRY_INTERVAL = 10
    cmd = './run_start.sh job' + job + ' K ' + testcase
    while True:
        stdout = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE).stdout
        stdout = str(stdout)      
        print(stdout)
        if 'Submitted batch job' in stdout:        
            break
        else:
            kill_job(job, 'K')
            print('encountered sbatch error on job start, retrying...')
        time.sleep(RETRY_INTERVAL)

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
        if job not in pc_job:
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
           

############### first clear finish status and recorded power of all jobs ####################

finish_dict = {}
with open('finish.json', 'r') as fp:
    finish_dict = json.load(fp)
for key in finish_dict:
    finish_dict[key] = 0
with open('finish.json', 'w') as fp:
    json.dump(finish_dict, fp)

power_dict = {}
with open('power.json', 'r') as fp:
    power_dict = json.load(fp)
for key in power_dict:
    power_dict[key] = 0
with open('power.json', 'w') as fp:
    json.dump(power_dict, fp)

######################################################################

while True:
    
    # termination condition: 
    # all the jobs have finished

    ################### check for finished jobs on K80 and V100 ##############################

    with open('finish.json', 'r') as fp:
        finish_dict = json.load(fp)

    for job in K80_job[:]:
        if finish_dict['job'+job] == 1:
            K80_used -= 1            
            K80_job.remove(job)
            print('K80 finished job: ' + job)
            JCT[job] = int(time.time() - job_start[job])   

    for job in V100_job[:]:
        if finish_dict['job'+job] == 1:
            V100_used -= 1            
            V100_job.remove(job)
            print('V100 finished job: ' + job)
            JCT[job] = int(time.time() - job_start[job])
    
    ################ check for practical finished jobs on K80 and V100 ######################

    all_job = K80_job + V100_job
    check_practical_complete(all_job)

    ################ check run time of current K80 job, update qualified_job #################

    with open('power.json', 'r') as fp:
        power_dict = json.load(fp)

    for job in K80_job:
        if job not in qualified_job:
            pwr_meas = power_dict['job'+job]
            if pwr_meas > 0:
                pdb.set_trace()
                qualified_job.append(job)
                print('job' + job + ' has been qualified for promotion')

    ################ make promotion decisions ########################

    V100_free = V100_cap - V100_used
    # this returns available jobs for promotion. Has to be qualified, and currently in K80, but not practically complete
    promote_list = list(set(qualified_job).intersection(K80_job).difference(pc_job))
    # this returns job forced to be demoted. Currently in V100, and is practically complete
    force_demote = list(set(V100_job).intersection(pc_job))

    if len(promote_list) > 0:
        #promoted, demoted = random_promotion(V100_free, promote_list, force_demote)
        promoted, demoted = max_param_promotion(V100_job, promote_list, force_demote) 
        if len(promoted) > 0:
            print('promoted jobs: ', promoted)
        if len(demoted) > 0:
            print('demoted jobs: ', demoted)
        # stop all promoted jobs on K80
        for job in promoted:
            scancel_job(job, 'K')
            print('K80 job canceled: ' + job)
            K80_job.remove(job)
            K80_used -= 1
        # stop all demoted jobs on V100
        for job in demoted:
            scancel_job(job, 'V')
            print('V100 job canceled: ' + job)
            V100_job.remove(job)
            V100_used -= 1

        # resume promoted jobs on V100
        for job in promoted:
            wait_till_job_ends(job, 'K')
            resume_job(job, 'V')
            print('V100 job resumed: ' + job)
            V100_job.append(job)
            V100_used += 1
        # resume demoted jobs on K80
        for job in demoted:
            wait_till_job_ends(job, 'V')
            resume_job(job, 'K')
            print('K80 job resumed: ' + job)
            K80_job.append(job)
            K80_used += 1

    ################ submit new jobs to vacant K80 GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
    if K80_used < K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            if index < len(queue):
                job = str(queue[index])
                start_job(job)
                print('new job created on K80: ' + job)
                K80_job.append(job)
                job_start[job] = time.time()
                index += 1
                K80_used += 1

    ############### wait for next iteration

    time.sleep(INTERVAL)

    ################ check if termination condition is met ################

    if len(K80_job) == 0 and len(V100_job) == 0 and index == len(queue):
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
