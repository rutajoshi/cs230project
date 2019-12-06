from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from PIL import Image
from scipy import ndimage
import pandas as pd
import imageio

# tensorflow stuff
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K

from datetime import datetime
from packaging import version

# constants
BATCH_SIZE = 25
IMG_HEIGHT = 855
IMG_WIDTH = 1280

DATASET_SIZE = 10000
TRAIN_SIZE = 6500
VAL_SIZE = 1500
TEST_SIZE = int(DATASET_SIZE * .2)

STEPS_PER_EPOCH = np.ceil(TRAIN_SIZE/BATCH_SIZE)

# first 4000 images
labels = pd.read_csv('labels.csv', float_precision='road_trip')
labels["base color R"] = labels["base color R"].str[1:]
labels["base color B"] = labels["base color B"].str[:-1]
labels["specular color R"] = labels["specular color R"].str[1:]
labels["specular color B"] = labels["specular color B"].str[:-1]
labels["img_name"] = labels["img_name"].str[:-3]+"jpg"

vector_values = labels.iloc[:,2:]
vector_values = vector_values.astype(float)
vector_values.head(5)

# rest of 6000 images
labels_rest = pd.read_csv('big_dataset.csv', float_precision='road_trip')
labels_rest["img_name"] = labels_rest["img_name"].str[:-3]+"jpg"
labels_rest.head(5)

vector_values2 = labels_rest.iloc[:,2:]
vector_values2 = vector_values2.astype(float)

# concat the 4000 and 6000 images for full dataset
frames = [vector_values, vector_values2]
vector_value = pd.concat(frames)
np_labels = vector_value.values

# START: generator helper functions

# Crop the input image to a bounding box and resize to fit VGG16 input size
def crop(img):
    # Note: image_data_format is 'channel_last'
    x = 144
    y = 59
    crop_size = 224
    return img[y:(y+crop_size), x:(x + crop_size), :]

# Generator function that is used to stream labels to keras
def label_gen(labels, batch_size):
    num = 0
    while True:
        yield labels[num: num + BATCH_SIZE, :]
        num += batch_size
        if (num >= len(labels)):
            num = 0

<<<<<<< HEAD
def crop_gen(batches, labels):
=======
# Generator function that is used to stream cropped batches and their labels to keras
# It takes in two generator functions: "batches" and "labels" which are queried for
# the next available batch of images and labels
def crop_generator(batches, labels, crop_length, batch_size, steps):
    i = 0
    lock = Lock()
>>>>>>> master
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros((BATCH_SIZE, 224, 224, 3))
        for i in range(BATCH_SIZE):
            batch_crops[i] = crop(batch_x[0][i])
<<<<<<< HEAD
        yield (batch_crops, next(labels))    
        
=======
        assert(labels_x.shape[0] == batch_crops.shape[0], "labels size = " + str(labels_x.shape[0]))
        yield (batch_crops, labels_x)

        lock.acquire()
        i = i + 1
        if (i >= steps):
            i = 0
        lock.release()

# This is the keras image generator that uses the above 2 functions to generate training/test batches
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


>>>>>>> master
# END: generator helper functions

# Creating image generators for each dataset (65/15/20 percent split)
# The flow_from_directory function returns a generator that streams images from target directories
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str("./data_split/train_data"),
                                                     target_size=(int(IMG_HEIGHT * .4), int(IMG_WIDTH * .4)),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
val_data_gen = image_generator.flow_from_directory(directory=str("./data_split/val_data"),
                                                     target_size=(int(IMG_HEIGHT * .4), int(IMG_WIDTH * .4)),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
test_data_gen = image_generator.flow_from_directory(directory=str("./data_split/test_data"),
                                                     target_size=(int(IMG_HEIGHT * .4), int(IMG_WIDTH * .4)),
                                                     batch_size=BATCH_SIZE,
                                                     classes = None)

# Creating label generators for each dataset (65/15/20 percent split)
# (Set up the label datasets to be passed into the generators)
Y_train = np_labels[:TRAIN_SIZE]
Y_val = np_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
Y_test = np_labels[TRAIN_SIZE + VAL_SIZE:]
Y_train_gen = label_gen(Y_train, BATCH_SIZE)
Y_val_gen = label_gen(Y_val, BATCH_SIZE)
Y_test_gen = label_gen(Y_test, BATCH_SIZE)

<<<<<<< HEAD
train_crops = crop_gen(train_data_gen, Y_train_gen)
val_crops = crop_gen(val_data_gen, Y_val_gen)
test_crops = crop_gen(test_data_gen, Y_test_gen)
=======
# Create the train data and test data batch/label pair generators using the
# corresponding generators for images streamed from a directory
train_crops = crop_generator(train_data_gen, Y_train_gen, 855, BATCH_SIZE, int(6500/BATCH_SIZE))
val_crops = crop_generator(val_data_gen, Y_val_gen, 855, BATCH_SIZE, int(1500/BATCH_SIZE))
test_crops = crop_generator(test_data_gen, Y_test_gen, 855, BATCH_SIZE, int(2000/BATCH_SIZE))

# Define the Keras TensorBoard callback
# Also make log directories based on timestamp
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
>>>>>>> master

# Get the pretrained VGG model. We set include_top to False to specify that we don't want the classification layers
initial_model = VGG16(weights='imagenet', include_top=False)
# Set all existing VGG16 layers to be untrainable (freeze weights)
for layer in initial_model.layers:
    layer.trainable = False

# ARCHITECTURE CHANGES ARE HERE

# Version 1 - this flattens the output and passes into a dense 10 layer output
preds = Flatten()(initial_model.output)
preds.set_shape((None, 25088))
preds = Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)

# Version 2 - this passes the output into:
# a conv layer, tanh activation, spatial dropout, flattens, and into a dense layer
# preds = Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
# preds = Activation('tanh')(preds)
# preds = SpatialDropout2D(0.4)(preds)
# preds = Flatten()(preds)
# preds.set_shape((None, 1470))
# preds = Dense(10, activation='sigmoid', input_shape=(None, 1470), trainable=True)(preds)

# Version 3 - this passes the output into:
# a conv layer, tanh activation, another conv layer, spatial dropout, flattens, and into a dense layer
# preds = Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
# preds = Activation('tanh')(preds)
# preds = Conv2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last') (preds)
# preds = SpatialDropout2D(0.4)(preds)
# preds = Flatten()(preds)
# preds.set_shape((None, 1296))
# preds = Dense(10, activation='sigmoid', input_shape=(None, 1296), trainable=True)(preds)


# Initialize and compile keras model
model = keras.Model(initial_model.input, preds)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Set the train and validation step sizes by calculating the number of
# batches in the train and test sets respectively
STEP_SIZE_TRAIN=int(6500/BATCH_SIZE)
STEP_SIZE_VALID=int(1500/BATCH_SIZE)

<<<<<<< HEAD
# Callback functions
# Define the Keras TensorBoard callback.
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('vgg16_best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# fit generator
=======
# Fit the model to the training data and validate on the validation data
>>>>>>> master
model.fit_generator(generator=train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_crops,
                    validation_steps= STEP_SIZE_VALID,
                    epochs=10000,
                    verbose=True,
                    use_multiprocessing = False,
<<<<<<< HEAD
                    max_queue_size=10, 
                    callbacks=[es, mc, tensorboard]
=======
                    max_queue_size=10,
                    workers=1,
                    callbacks=[tensorboard]
>>>>>>> master
)

STEP_SIZE_TEST=int(2000/BATCH_SIZE)

# Re-evaluate the model on the test data
loss, acc = model.evaluate_generator(generator = test_crops, steps = STEP_SIZE_TEST, max_queue_size = 10, workers = 1, callbacks = [])
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Restored model, loss: {:5.2f}".format(loss))
<<<<<<< HEAD
=======

# Save the weights in a checkpoint
model.save_weights('./checkpoints/{time}')
>>>>>>> master
