#!/bin/bash
echo "run nvidia-smi command to monitor gpu power"

RUN_TIME=100
TESTCASE=$1  #"resnet50_32"
DATA_PATH="/home/gridsan/baolinli/logs/GPU_pwr_meas/tensorflow/"

mkdir -p ${DATA_PATH}${TESTCASE}

timeout ${RUN_TIME} nvidia-smi -i 0 --query-gpu=index,timestamp,power.draw,memory.used,utilization.memory,utilization.gpu,temperature.gpu --format=csv,nounits -lms 10 --filename=${DATA_PATH}${TESTCASE}/sample.csv

