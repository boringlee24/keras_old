"""
#Trains a ResNet on the CIFAR10 dataset.

"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras import models, layers, optimizers
from datetime import datetime
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import os
import pdb
import sys
import argparse
import time
import signal
import glob
import json
import send_signal
import pathlib
from scipy.stats import variation
import math

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='specific testcase name')
parser.add_argument('--resume', dest='resume', action='store_true', help='if True, resume training from a checkpoint')
parser.add_argument('--gpu_num', metavar='GPU_NUMBER', type=str, help='select which gpu to use')
parser.add_argument('--node', metavar='HOST_NODE', type=str, help='node of the host (scheduler)')
parser.set_defaults(resume=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num

# Training parameters
batch_size = 256
args_lr = 0.002
args_model = 'densenet201'

epoch_begin_time = 0

job_name = sys.argv[0].split('.')[0]
save_files = '/scratch/li.baol/dl_checkpoints/' + args.tc + '/' + job_name + '_*'

total_epochs = 36
starting_epoch = 0

# first step is to update the PID
pid = os.getpid()
message = job_name + ' pid ' + str(pid) # 'job50 pid 3333'
send_signal.send(args.node, 10002, message)

if args.resume:
    save_file = glob.glob(save_files)[0]
#    epochs = int(save_file.split('/')[4].split('_')[1].split('.')[0])
    starting_epoch = int(save_file.split('/')[5].split('.')[0].split('_')[-1])

data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

n = 3

# Model name, depth and version
model_type = args.tc #'P100_resnet50_he_256_1'

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

with tf.device('/cpu:0'):
    if args.resume:
        print('resume from checkpoint')
        message = job_name + ' b_end'
        send_signal.send(args.node, 10002, message)
        model = keras.models.load_model(save_file)
        message = job_name + ' c_end'
        send_signal.send(args.node, 10002, message)
    else:
        print('train from start')
        model = models.Sequential()
        
        if '121' in args_model:
            base_model = DenseNet121(weights=None, include_top=False, input_shape=(32, 32, 3), pooling='avg')
        elif '169' in args_model:
            base_model = DenseNet169(weights=None, include_top=False, input_shape=(32, 32, 3), pooling='avg')
        elif '201' in args_model:
            base_model = DenseNet201(weights=None, include_top=False, input_shape=(32, 32, 3), pooling='avg')
            
           
        model.add(base_model)
        #model.add(layers.Flatten())
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dense(128, activation='relu'))
        #model.add(layers.Dropout(0.5))
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dropout(0.5))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dense(10, activation='softmax'))#, kernel_initializer='he_uniform'))

parallel_model = multi_gpu_model(model, gpus=2, cpu_merge=True)

parallel_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=args_lr),
              metrics=['accuracy'])

#model.summary()
print(model_type)

#pdb.set_trace()

batch_time = []
batch_begin = 0

################### connects interrupt signal to the process #####################

def terminateProcess(signalNumber, frame):
    # first record the wasted epoch time
    global epoch_begin_time
    if epoch_begin_time == 0:
        epoch_waste_time = 0
    else:
        epoch_waste_time = int(time.time() - epoch_begin_time)

    message = job_name + ' waste ' + str(epoch_waste_time) # 'job50 waste 100'
    if epoch_waste_time > 0:
        send_signal.send(args.node, 10002, message)

    print('checkpointing the model triggered by kill -15 signal')
    # delete whatever checkpoint that already exists
    for f in glob.glob(save_files):
        os.remove(f)
    pathlib.Path('/scratch/li.baol/dl_checkpoints/'+args.tc+'/').mkdir(parents=True, exist_ok=True)
    model.save('/scratch/li.baol/dl_checkpoints/'+args.tc+'/' + job_name + '_' + str(current_epoch) + '.h5')
    print ('(SIGTERM) terminating the process')

    message = job_name + ' checkpoint'
    send_signal.send(args.node, 10002, message)

    sys.exit()

signal.signal(signal.SIGTERM, terminateProcess)

#################################################################################

logdir = '/scratch/li.baol/tsrbrd_log/job_runs/' + model_type + '/' + job_name

tensorboard_callback = TensorBoard(log_dir=logdir)#, update_freq='batch')

first_epoch_start = 0
batches_per_epoch = math.ceil(y_train.shape[0] / batch_size)
stable_batch = 0

class PrintEpoch(keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs=None):
        global batch_begin
        batch_begin = time.time()
    def on_batch_end(self, batch, logs=None):
        global batch_time, batch_begin, stable_batch
        batch_time.append(float(time.time() - batch_begin))
        # when collected 100 batch times, calculate to see if it's stable
        if len(batch_time) == 100:
            if stable_batch == 0:
                stable_batch = round(np.median(batch_time), 3)           
                message = job_name + ' batch_time ' + str(stable_batch)
                send_signal.send(args.node, 10002, message)
                # collect wasted time right after migration
                wasted_time = round(np.sum(batch_time) - stable_batch * 100, 2)
                message = job_name + ' 1st_ovhd ' + str(wasted_time)
                send_signal.send(args.node, 10002, message)
            batch_time = []
            self.remaining_batches -= 100
            message = job_name + ' remain_batch ' + str(self.remaining_batches)
            send_signal.send(args.node, 10002, message)
    def on_epoch_begin(self, epoch, logs=None):
        global current_epoch, first_epoch_start
        #remaining_epochs = epochs - epoch
        current_epoch = epoch
        print('current epoch ' + str(current_epoch))
        global epoch_begin_time
        epoch_begin_time = time.time()
        if epoch == starting_epoch and args.resume:
            first_epoch_start = time.time()
            message = job_name + ' d_end'
            send_signal.send(args.node, 10002, message)
        elif epoch == starting_epoch:
            first_epoch_start = time.time()           
        if epoch == starting_epoch:
            # send signal to indicate checkpoint is qualified
            message = job_name + ' ckpt_qual'
            send_signal.send(args.node, 10002, message)
            self.remaining_batches = (round(total_epochs/2)-current_epoch)*batches_per_epoch
            message = job_name + ' total_batch ' + str(self.remaining_batches)
            send_signal.send(args.node, 10002, message)
        message = job_name + ' epoch_begin ' + str(current_epoch)
        send_signal.send(args.node, 10002, message)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == starting_epoch:
            first_epoch_time = int(time.time() - first_epoch_start)
            message = job_name + ' 1st_epoch ' + str(first_epoch_time)
            send_signal.send(args.node, 10002, message)
        progress = round((epoch+1) / round(total_epochs/2), 2)
        message = job_name + ' completion ' + str(progress)
        send_signal.send(args.node, 10002, message)

my_callback = PrintEpoch()

callbacks = [tensorboard_callback, my_callback]
 #[checkpoint, lr_reducer, lr_scheduler, tensorboard_callback]

# Run training

parallel_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=round(total_epochs/2),
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks,
          initial_epoch=starting_epoch,
          verbose=1
          )

# Score trained model.
scores = parallel_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# send signal to indicate job has finished
message = job_name + ' finish'
send_signal.send(args.node, 10002, message)
