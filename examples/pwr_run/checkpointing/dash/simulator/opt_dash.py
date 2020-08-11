import numpy as np
import cvxpy as cp

def optimize_promotion(num_GPUs, job_num_GPUs, job_remaining_time):

    num_GPU_types = len(num_GPUs)

    jobs = list(job_num_GPUs.keys())

    num_jobs = len(jobs)

    num_variables = num_GPU_types * num_jobs

    num_GPUs_list = [job_num_GPUs[jobs[j]] for j in range(num_jobs)]

    remaining_times = []
    for j in range(num_jobs):
        for r in job_remaining_time[jobs[j]]:
            remaining_times.append(r*num_GPUs_list[j])
    remaining_times = np.matrix(remaining_times)

    x = cp.Variable(num_variables, boolean=True)

    objective = cp.Minimize(remaining_times*x) #TODO

    constraints = []
    for j in range(num_jobs):
        constraints.append(
            sum(x[j*num_GPU_types:j*num_GPU_types+num_GPU_types]) == 1)

    for g in range(num_GPU_types):
        indices = range(g, num_variables, num_GPU_types)
        constraints.append(np.matrix(num_GPUs_list)*x[indices] <= num_GPUs[g])

    problem = cp.Problem(objective, constraints)

    problem.solve()

    job_GPU_types = {}
    for j in range(num_jobs):
        gpu_type = np.where(
            x.value[j*num_GPU_types:j*num_GPU_types+num_GPU_types] == 1)[0][0]
        job_GPU_types[jobs[j]] = gpu_type

    return job_GPU_types

def max_speedup_first(num_GPUs, job_num_GPUs, job_remaining_time):
    job_speedup = {}
    for job in job_remaining_time:
        job_speedup[job] = job_remaining_time[job][0] / job_remaining_time[job][1]
    num_V100 = num_GPUs[1]
    num_jobs = len(job_num_GPUs)
    sorted_pool = sorted(job_speedup, key=job_speedup.get, reverse=True)[:num_V100] 
    jobs = list(job_num_GPUs.keys())

    job_GPU_types = {}
    for i in range(num_jobs):
        job_GPU_types[jobs[i]] = 1 if jobs[i] in sorted_pool else 0
    return job_GPU_types


def shortest_jobs_first(num_GPUs, job_num_GPUs, job_remaining_time):
    v100_jrt = {}
    for job in job_remaining_time:
        v100_jrt[job] = job_remaining_time[job][1]
    num_V100 = num_GPUs[1]
    num_jobs = len(job_num_GPUs)
    sorted_pool = sorted(v100_jrt, key=v100_jrt.get, reverse=False)[:num_V100]
    jobs = list(job_num_GPUs.keys())

    job_GPU_types = {}
    for i in range(num_jobs):
        job_GPU_types[jobs[i]] = 1 if jobs[i] in sorted_pool else 0
    return job_GPU_types


#num_GPUs = [3, 1]  # K80, #P100, #V100 (order doesn't matter)
#job_num_GPUs = {'1': 1, '3': 1, '4': 1}#, '17': 1, '30': 1}#,
##                '45': 1, '60': 3}  # number of GPUs that the job is running on
#job_remaining_time = {'1': [20000, 10000], '3': [3000, 1000], '4': [500, 0]}#, '17': [400, 300], '30': [
##    100, 0]}#, '45': [400, 300], '60': [600, 500]}  # same order of runtimes as in 'num_GPUs'
#
## GPU id in the same order as 'num_GPUs'
#job_GPU_types = shortest_jobs_first(num_GPUs, job_num_GPUs, job_remaining_time)
#
#print(job_GPU_types)
