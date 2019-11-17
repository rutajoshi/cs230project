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
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import TensorBoard

from datetime import datetime
from packaging import version

%matplotlib inline


# Load labels from csv
labels = pd.read_csv('labels.csv', float_precision='road_trip')
# Clean labels dataset
labels["base color R"] = labels["base color R"].str[1:]
labels["base color B"] = labels["base color B"].str[:-1]
labels["specular color R"] = labels["specular color R"].str[1:]
labels["specular color B"] = labels["specular color B"].str[:-1]
labels["img_name"] = labels["img_name"].str[:-3]+"jpg"
vector_values = labels.iloc[:,2:]
vector_values = vector_values.astype(float)
np_labels = vector_values.values

# Set Constants
BATCH_SIZE = 40
IMG_HEIGHT = 855
IMG_WIDTH = 1280
STEPS_PER_EPOCH = np.ceil(4000/BATCH_SIZE)

# Crop the input image to a bounding box and resize to fit VGG16 input size
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
        yield labels[num: num + BATCH_SIZE, :]
        num += batch_size
        if (num >= len(labels)):
            num = 0

# Generator function that is used to stream cropped batches and their labels to keras
# It takes in two generator functions: "batches" and "labels" which are queried for
# the next available batch of images and labels
def crop_generator(batches, labels, crop_length):
    while True:
        batch_x = next(batches)
        labels_x = next(labels)
        start_y = (855 - crop_length) // 2
        start_x = (1280 - crop_length) // 2
        batch_crops = np.zeros((BATCH_SIZE, 224, 224, 3))
        for i in range(batch_crops.shape[0]):
            batch_crops[i] = crop(batch_x[0][i])
        yield (batch_crops, labels_x)

# This is the keras image generator that uses the above 2 functions to generate training/test batches
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Use a 90/10 training/test split and generate training and test batches
# The flow_from_directory function returns a generator that streams images from target directories
train_data_gen = image_generator.flow_from_directory(directory=str("./data_split/train_data"),
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
test_data_gen = image_generator.flow_from_directory(directory=str("./data_split/test_data"),
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=400,
                                                     classes = None)

# Set up the label datasets to be passed into the generators
Y_train = np_labels[:3600]
Y_test = np_labels[3600:]
Y_train_gen = label_gen(Y_train, BATCH_SIZE)
Y_test_gen = label_gen(Y_test, 400)

# Get the pretrained VGG model. We set include_top to False to specify that we don't want the classification layers
initial_model = VGG16(weights='imagenet', include_top=False)

# Freeze the weights of the pretrained model
for layer in initial_model.layers:
    layer.trainable = False

# Flatten the output of the VGG (a feature map) to pass into new architecture:
# We've added the following new layers:
#   - Dense output layer of 10 sigmoid neurons
preds = layers.Flatten()(initial_model.output)
preds.set_shape((None, 25088))
preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)

# Initialize and compile keras model
model = keras.Model(initial_model.input, preds)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Create the train data and test data batch/label pair generators using the
# corresponding generators for images streamed from a directory
train_crops = crop_generator(train_data_gen, Y_train_gen, 855)
test_crops = crop_generator(test_data_gen, Y_test_gen, 855)

# Define the Keras TensorBoard callback
# Also make log directories based on timestamp
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # added
log_dir = f"logs/{time}" # added
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Set the train and validation step sizes by calculating the number of
# batches in the train and test sets respectively
STEP_SIZE_TRAIN=train_data_gen.n//train_data_gen.batch_size
STEP_SIZE_VALID=test_data_gen.n//test_data_gen.batch_size

# Fit the model to the training data and validate on the "test" data
model.fit_generator(generator=train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_crops,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=[tensorboard]
)
