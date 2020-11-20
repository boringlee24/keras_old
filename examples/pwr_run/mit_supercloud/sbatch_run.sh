#export NETWORK='resnet'
#export ARCH='resnet50'
#export BATCH='32'
#export LR='0.001'
#export TESTCASE=${ARCH}_${BATCH}
#sbatch sbatch.sh
#
#export NETWORK='vgg'
#export ARCH='vgg19'
#export BATCH='32'
#export LR='0.001'
#export TESTCASE=${ARCH}_${BATCH}
#sbatch sbatch.sh
#
#export NETWORK='densenet'
#export ARCH='densenet121'
#export BATCH='32'
#export LR='0.001'
#export TESTCASE=${ARCH}_${BATCH}
#sbatch sbatch.sh
#
#export NETWORK='resnet'
#export ARCH='resnet50'
#export BATCH='256'
#export LR='0.001'
#export TESTCASE=${ARCH}_${BATCH}
#sbatch sbatch.sh
#
export NETWORK='vgg'
export ARCH='vgg19'
export BATCH='256'
export LR='0.001'
export TESTCASE=${ARCH}_${BATCH}
sbatch sbatch.sh
#
#export NETWORK='densenet'
#export ARCH='densenet121'
#export BATCH='256'
#export LR='0.001'
#export TESTCASE=${ARCH}_${BATCH}
#sbatch sbatch.sh

