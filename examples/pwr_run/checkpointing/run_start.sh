export MYJOB=$1
sbatch --job-name=${MYJOB} job_start_${2}.sh
