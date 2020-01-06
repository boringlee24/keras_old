#!/bin/bash

# arg1: jobx. arg2: GPU. arg3: testcase
export MYJOB=$1
export TESTCASE=$3
sbatch --job-name=${MYJOB}_${2} job_resume_${2}.sh
