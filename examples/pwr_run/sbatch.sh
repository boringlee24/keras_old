#!/bin/bash

#SBATCH --job-name=tensorflow                    # sets the job name
#SBATCH --tasks-per-node=1                        # sets 2 tasks for each machine
#SBATCH --nodes=1
#SBATCH --mem=20Gb                               # reserves 100 GB memory
#SBATCH --partition=multigpu                  # requests that the job is executed in partition my partition
##SBATCH --reservation=hpc2020
#SBATCH --output=/scratch/li.baol/slurm/%j.out               # sets the standard output to be stored in file my_nice_job.%j.out,
#SBATCH --error=/scratch/li.baol/slurm/%j.err                # sets the standard error to be stored in file my_nice_job.%j.err,
#SBATCH --gres=gpu:v100-sxm2:1                              # reserves 1 gpu per machine
#SBATCH --exclude=c[2184,2192],d[1013,1005,1001]   #[2204-2207]
#SBATCH --time=24:0:0

#srun ./run.sh $TESTCASE & python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr 0.001
srun python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr $LR

