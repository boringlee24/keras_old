#!/bin/bash

#SBATCH --job-name=abc                    # sets the job name
#SBATCH --tasks-per-node=1                        # sets 2 tasks for each machine
#SBATCH --nodes=1
#SBATCH --mem=50Gb                               # reserves 100 GB memory
#SBATCH --partition=gpu                  # requests that the job is executed in partition my partition
#SBATCH --output=/scratch/li.baol/slurm/%j.out               # sets the standard output to be stored in file my_nice_job.%j.out,
#SBATCH --error=/scratch/li.baol/slurm/%j.err                # sets the standard error to be stored in file my_nice_job.%j.err,
#SBATCH --gres=gpu:v100-sxm2:1                              # reserves 1 gpu per machine
#SBATCH --exclude=c[2184,2192],d[1005,1013]   #[2204-2207]

srun python ${MYJOB}.py --tc ${MYJOB} --model resnet50 -b 32 --lr 0.001 --resume
