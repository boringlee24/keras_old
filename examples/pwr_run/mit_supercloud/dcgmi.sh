#!/bin/bash

dcgmi stats -x ${SLURM_JOBID} -v 

dcgmi dmon -d 1000 -e 203,204,210,211,1002,1003,1009,1010 > ~/csv/dcgmi-dmon-${SLURM_JOBID}.csv

