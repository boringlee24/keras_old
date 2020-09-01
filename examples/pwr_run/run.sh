#!/bin/bash
echo "run nvidia-smi command to monitor gpu power"

# run for 2s, sample every 10s, 2hrs in total (7200 s)
RUN_TIME=2
SAMPLING_INTERVAL=8
NUM_SAMPLES=720
SAMPLE=0
TESTCASE=$1  #"P100_noshuffle_resnet152"
DATA_PATH="/scratch/li.baol/GPU_pwr_meas/tensorflow/"

mkdir -p ${DATA_PATH}${TESTCASE}

while [ $SAMPLE -lt $NUM_SAMPLES ]
do
    timeout ${RUN_TIME} nvidia-smi -i 0 --query-gpu=index,timestamp,power.draw,memory.used,utilization.memory,utilization.gpu,temperature.gpu --format=csv,nounits -lms 10 --filename=${DATA_PATH}${TESTCASE}/sample_${SAMPLE}.csv
    sleep $SAMPLING_INTERVAL
    ((SAMPLE++))
done
