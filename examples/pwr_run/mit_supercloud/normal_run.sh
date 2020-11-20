
export NETWORK='vgg'
export ARCH='vgg19'
export BATCH='256'
export LR='0.001'
export TESTCASE=${ARCH}_${BATCH}
python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr $LR && ./dcgmi_collect.sh
#./dcgmi.sh & python ${NETWORK}.py --tc $TESTCASE --model $ARCH -b $BATCH --lr $LR
