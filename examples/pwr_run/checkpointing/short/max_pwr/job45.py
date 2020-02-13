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
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import models, layers, optimizers
from datetime import datetime
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

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('--tc', metavar='TESTCASE', type=str, help='specific testcase name')
parser.add_argument('--resume', dest='resume', action='store_true', help='if True, resume training from a checkpoint')
parser.add_argument('--gpu_num', metavar='GPU_NUMBER', type=str, help='select which gpu to use')
parser.set_defaults(resume=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num

# Training parameters
batch_size = 32
args_lr = 0.003

epoch_begin_time = 0

job_name = sys.argv[0].split('.')[0]
save_files = '/scratch/li.baol/checkpoint_max_pwr/' + job_name + '*'

total_epochs = 95
starting_epoch = 0

# first step is to update the PID
pid_dict = {}
with open('pid_lock.json', 'r') as fp:
    pid_dict = json.load(fp)
pid_dict[job_name] = os.getpid()
json_file = json.dumps(pid_dict)
with open('pid_lock.json', 'w') as fp:
    fp.write(json_file) 
os.rename('pid_lock.json', 'pid.json')

if args.resume:
    save_file = glob.glob(save_files)[0]
#    epochs = int(save_file.split('/')[4].split('_')[1].split('.')[0])
    starting_epoch = int(save_file.split('/')[4].split('.')[0].split('_')[-1])

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

if args.resume:
    print('resume from checkpoint')
    model = keras.models.load_model(save_file)
else:
    print('train from start')
    model = models.Sequential()
    
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)
    
    #base_model.summary()
    
    #pdb.set_trace()
    
    model.add(base_model)
    model.add(layers.Flatten())
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=args_lr),
                  metrics=['accuracy'])
    
    #model.summary()
    print(model_type)

#pdb.set_trace()

current_epoch = 0

################### connects interrupt signal to the process #####################

def terminateProcess(signalNumber, frame):
    # first record the wasted epoch time
    global epoch_begin_time
    epoch_waste_time = int(time.time() - epoch_begin_time)

    epoch_waste_dict = {}
    with open('epoch_waste.json', 'r') as fp:
        epoch_waste_dict = json.load(fp)
    epoch_waste_dict[job_name] += epoch_waste_time
    json_file3 = json.dumps(epoch_waste_dict)
    with open('epoch_waste.json', 'w') as fp:
        fp.write(json_file3)

    print('checkpointing the model triggered by kill -15 signal')
    # delete whatever checkpoint that already exists
    for f in glob.glob(save_files):
        os.remove(f)
    model.save('/scratch/li.baol/checkpoint_max_pwr/' + job_name + '_' + str(current_epoch) + '.h5')
    print ('(SIGTERM) terminating the process')

    checkpoint_dict = {}
    with open('checkpoint.json', 'r') as fp:
        checkpoint_dict = json.load(fp)
    checkpoint_dict[job_name] = 1
    json_file3 = json.dumps(checkpoint_dict)
    with open('checkpoint.json', 'w') as fp:
        fp.write(json_file3)

    sys.exit()

signal.signal(signal.SIGTERM, terminateProcess)

#################################################################################

logdir = '/scratch/li.baol/tsrbrd_log/job_runs/' + model_type + '/' + job_name

tensorboard_callback = TensorBoard(log_dir=logdir)#, update_freq='batch')

class PrintEpoch(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        global current_epoch 
        #remaining_epochs = epochs - epoch
        current_epoch = epoch
        print('current epoch ' + str(current_epoch))
        global epoch_begin_time
        epoch_begin_time = time.time()

my_callback = PrintEpoch()

callbacks = [tensorboard_callback, my_callback]
 #[checkpoint, lr_reducer, lr_scheduler, tensorboard_callback]

# Run training

# creates an file if job qualified for checkpoint
open('ckpt_qual/' + job_name + '.txt', 'a').close()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=round(total_epochs/2),
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks,
          initial_epoch=starting_epoch,
          verbose=1
          )

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

finish_dict = {}
while True:
    if os.path.exists('finish.json'):
        try:
            os.rename('finish.json', 'finish_lock.json')
            break
        except Exception:
            pass
    else:
        time.sleep(1)
with open('finish_lock.json', 'r') as fp:
    finish_dict = json.load(fp)
finish_dict[job_name] = 1
json_file2 = json.dumps(finish_dict)
with open('finish_lock.json', 'w') as fp:
    fp.write(json_file2)
os.rename('finish_lock.json', 'finish.json')
