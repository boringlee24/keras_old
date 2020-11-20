#!/bin/bash

dcgmi stats -j ${SLURM_JOBID} -v > ~/csv/dcgmi-stats-${SLURM_JOBID}.csv 

