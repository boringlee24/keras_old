#!/bin/bash

#SBATCH --job-name=tensorflow                    # sets the job name
#SBATCH --nodes=1
##SBATCH --mem=20Gb                               # reserves 100 GB memory
#SBATCH --partition=normal                  # requests that the job is executed in partition my partition
#SBATCH --output=/home/gridsan/baolinli/logs/slurm/%j.out               # sets the standard output to be stored in file my_nice_job.%j.out,
#SBATCH --error=/home/gridsan/baolinli/logs/slurm/%j.err                # sets the standard error to be stored in file my_nice_job.%j.err,
#SBATCH --gres=gpu:volta:1                              # reserves 1 gpu per machine
#SBATCH --constraint=xeon-g6
#SBATCH --cpus-per-task=20
#SBATCH --qos=high

#srun ./run.sh $TESTCASE & python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr 0.001
srun python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr $LR

