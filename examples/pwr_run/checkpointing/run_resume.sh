export MYJOB=$1
sbatch --job-name=${MYJOB} job_resume_${2}.sh
