#!/bin/bash
#echo "run nvidia-smi command to monitor gpu power"

JOB=$1  #"job1"
GPU=$2
TESTCASE=$3
DATA_PATH="/scratch/li.baol/GPU_pwr_meas/tensorflow/job_runs/${TESTCASE}/"
mkdir -p $DATA_PATH

# step 1: start power measurement
# step 2: after measurement done, process data, write to power.json
# in main.py, check if the power exist, if yes, job qualified for promote

nvidia-smi -i ${GPU} --query-gpu=index,timestamp,power.draw,memory.used,utilization.memory,utilization.gpu,temperature.gpu --format=csv,nounits -lms 100 --filename=${DATA_PATH}${JOB}.csv 

#sleep 305 && python gpu_pwr.py $JOB

#mv ${DATA_PATH}${JOB}.csv ${DATA_PATH}${JOB}_finish.csv

#while [ $SAMPLE -lt $NUM_SAMPLES ]
#do
#    timeout ${RUN_TIME} nvidia-smi -i 0 --query-gpu=index,timestamp,power.draw,memory.used,utilization.memory,utilization.gpu,temperature.gpu --format=csv,nounits -lms 10 --filename=${DATA_PATH}${JOB}/sample_${SAMPLE}.csv
#    sleep $SAMPLING_INTERVAL
#    ((SAMPLE++))
#done
