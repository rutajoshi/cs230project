from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from threading import Lock, Thread

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
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers

from datetime import datetime
from packaging import version
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.resnet import ResNet50

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

# threadsafe generator wrapper
class thread_safe_generator(object):
    def __init__(self, gen):
        self.gen = gen
        self.lock = Lock()

    def next(self):
        with self.lock:
            return next(self.gen)

# START: generator helper functions

# Crop the input image to a bounding box and resize to fit ResNet input size
def crop(img):
    # Note: image_data_format is 'channel_last'
    x = 212
    y = 0
    crop_size = 855
    new_img = tf.image.crop_to_bounding_box(img, 0, 212, crop_size, crop_size)
    return tf.image.resize(new_img, [224, 224])

# Generator function that is used to stream labels to keras
def label_gen(labels, batch_size):
    num = 0
    while True:
        if (num >= len(labels)):
            num = 0
        if (num == 0 and num + batch_size >= len(labels)):
            yield labels;
        else:
            yield labels[num: num + batch_size, :]
        num += batch_size

# Generator function that is used to stream cropped batches and their labels to keras
# It takes in two generator functions: "batches" and "labels" which are queried for
# the next available batch of images and labels
def crop_generator(batches, labels, crop_length, batch_size, steps):
    i = 0
    lock = Lock()
    while True:
        batch_x = batches[i]
        labels_x = next(labels)
        start_y = (855 - crop_length) // 2
        start_x = (1280 - crop_length) // 2
        batch_crops = np.zeros((batch_size, 224, 224, 3))
        for i in range(batch_crops.shape[0]):
            batch_crops[i] = crop(batch_x[0][i])
        assert(labels_x.shape[0] == batch_crops.shape[0], "labels size = " + str(labels_x.shape[0]))
        yield (batch_crops, labels_x)

        lock.acquire()
        i = i + 1
        if (i >= steps):
            i = 0
        lock.release()

# This is the keras image generator that uses the above 2 functions to generate training/test batches
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# END: generator helper functions

# Creating image generators for each dataset (65/15/20 percent split)
# The flow_from_directory function returns a generator that streams images from target directories
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str("./data_split_large/train_data"),
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
val_data_gen = image_generator.flow_from_directory(directory=str("./data_split_large/val_data"),
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
test_data_gen = image_generator.flow_from_directory(directory=str("./data_split_large/test_data"),
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
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

# Get the pretrained VGG model. We set include_top to False to specify that we don't want the classification layers
initial_model = ResNet50(weights='imagenet', include_top=False)
# Set all existing ResNet layers to be untrainable (freeze weights)
for layer in initial_model.layers:
    layer.trainable = False

# ARCHITECTURE CHANGES ARE HERE

# Version 1 - this flattens the output and passes into a dense 10 layer output
preds = layers.Flatten()(initial_model.output)
preds.set_shape((None, 25088))
preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)

# Version 2 - this passes the output into:
# a conv layer, tanh activation, spatial dropout, flattens, and into a dense layer
# preds = layers.Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
# preds = layers.Activation('tanh')(preds)
# preds = layers.SpatialDropout2D(0.4)(preds)
# preds = layers.Flatten()(preds)
# preds.set_shape((None, 1470))
# preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 1470), trainable=True)(preds)

# Version 3 - this passes the output into:
# a conv layer, tanh activation, another conv layer, spatial dropout, flattens, and into a dense layer
# preds = layers.Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
# preds = layers.Activation('tanh')(preds)
# preds = layers.Conv2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last') (preds)
# preds = layers.SpatialDropout2D(0.4)(preds)
# preds = layers.Flatten()(preds)
# preds.set_shape((None, 1296))
# preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 1296), trainable=True)(preds)


# Initialize and compile keras model
model = keras.Model(initial_model.input, preds)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Set the train and validation step sizes by calculating the number of
# batches in the train and test sets respectively
STEP_SIZE_TRAIN=int(6500/BATCH_SIZE)
STEP_SIZE_VALID=int(1500/BATCH_SIZE)

# Fit the model to the training data and validate on the validation data
model.fit_generator(generator=train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_crops,
                    validation_steps= STEP_SIZE_VALID,
                    epochs=10,
                    verbose=True,
                    use_multiprocessing = False,
                    max_queue_size=10,
                    workers=1,
                    callbacks=[tensorboard]
)

STEP_SIZE_TEST=int(2000/BATCH_SIZE)

# Re-evaluate the model on the test data
loss, acc = model.evaluate_generator(generator = test_crops, steps = STEP_SIZE_TEST, max_queue_size = 10, workers = 1, callbacks = [])
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Restored model, loss: {:5.2f}".format(loss))

# Save the weights in a checkpoint
model.save_weights('./checkpoints/{time}')
