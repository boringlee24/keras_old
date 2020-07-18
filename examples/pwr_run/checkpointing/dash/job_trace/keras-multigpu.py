import tensorflow as tf
from tensorflow import keras
import pdb
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.datasets import cifar10
from keras import models, layers, optimizers
from keras.optimizers import Adam
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2", "/gpu:3"])
#cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = models.Sequential()
    
    base_model = DenseNet201(weights=None, include_top=False, input_shape=(32, 32, 3), pooling='avg')
        
       
    model.add(base_model)
    model.add(layers.Dense(10, activation='softmax'))#, kernel_initializer='he_uniform'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

# Train the model on all available devices.
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.fit(x_train, y_train,
          batch_size=256,
          epochs=20,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1
          )

scores = model.evaluate(x_test, y_test, verbose=1)
