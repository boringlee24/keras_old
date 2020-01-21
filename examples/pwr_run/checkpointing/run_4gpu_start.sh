export MYJOB=$1
sbatch --job-name=${MYJOB} job_4gpu_start_${2}.sh
