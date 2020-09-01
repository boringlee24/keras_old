export NETWORK='mobilenet'
export ARCH='mobilenet'
export LR='0.001'

export BATCH='64'
export TESTCASE='V100_mobilenet_64'
sbatch sbatch.sh

export BATCH='128'
export TESTCASE='V100_mobilenet_128'
sbatch sbatch.sh

export BATCH='256'
export TESTCASE='V100_mobilenet_256'
sbatch sbatch.sh

export BATCH='32'
export TESTCASE='V100_mobilenet_32'
sbatch sbatch.sh

#export LR='0.05'
#export TESTCASE='V100_mobilenet_0.05'
#sbatch sbatch.sh
#
#export LR='0.01'
#export TESTCASE='V100_mobilenet_0.01'
#sbatch sbatch.sh
#
#export LR='0.005'
#export TESTCASE='V100_mobilenet_0.005'
#sbatch sbatch.sh
#
#export LR='0.001'
#export TESTCASE='V100_mobilenet_0.001'
#sbatch sbatch.sh
#
#export LR='0.0005'
#export TESTCASE='V100_mobilenet_0.0005'
#sbatch sbatch.sh

