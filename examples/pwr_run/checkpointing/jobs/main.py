import pdb
import time
import os
import subprocess
import re

queue = [49, 15, 50, 39, 14, 40, 13, 37, 32, 44, 1, 25, 6, 12, 43, 35, 29, 7, 46, 23, 47, 34, 21, 33, 36, 24, 28, 48, 17, 8, 45, 30, 2, 41, 16, 3, 27, 20, 38, 11, 42, 10, 22, 4, 18, 19, 5, 9, 26, 31]
JCT = []
index = 0

K80_cap = 8
V100_cap = 4
K80_used = 0
V100_used = 0
qualified_jobs = 0

K80_job = []
V100_job = []

testcase = 'K80_only'

INTERVAL = 60 # make decision every 30s

while True:
    
    # termination condition: 
    # all the jobs have finished

    ################### check for finished jobs ##############################

    # check list of currently running K80 jobs
    ##K80_out = subprocess.Popen(['squeue -u $USER | awk \"/_K/\" | awk \'{print $3}\''], 
    ##        stdout=subprocess.PIPE, 
    ##        stderr=subprocess.STDOUT,
    ##        shell=True)
    ##stdout,stderr = K80_out.communicate()
    ##stdout = str(stdout)
    stdout = subprocess.check_output(['squeue -u $USER | awk \"/_K/\" | awk \'{print $3}\''], shell=True)
    stdout = str(stdout)
    K80_run = re.findall(r'\d+', stdout) # this is list of currently running jobs in string
    # now confirm if there are jobs that are finished
    for job in K80_job[:]:
        if job not in K80_run:
            K80_used -= 1
            K80_job.remove(job)

    ################ check run time of current K80 jobs #################




    ################ submit jobs to vacant GPUs ############################

    # check if there are vacant K80s
    ## yes: submit jobs from queue
    ## no: do nothing
    if K80_used != K80_cap:
        K80_free = K80_cap - K80_used
        for i in range(K80_free):
            if index < len(queue):
                job = 'job' + str(queue[index])
                cmd = './run_start.sh ' + job + ' K ' + testcase
#                os.system(cmd)
                subprocess.call([cmd], shell=True)
                K80_job.append(str(queue[index]))           
                index += 1
                K80_used += 1




    ################ check if termination condition is met ################

    if len(K80_job) == 0 and index == len(queue):
        break

    # this loop performs actions every 30s
    time.sleep(INTERVAL)

print('finished all runs')
