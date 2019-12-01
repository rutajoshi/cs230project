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

def crop(img):
    # Note: image_data_format is 'channel_last'
    x = 212
    y = 0
    crop_size = 855
    return tf.image.crop_to_bounding_box(img, 0, 212, crop_size, crop_size)

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

    
def crop_generator(batches, labels, crop_length, batch_size, steps):
    i = 0
    lock = Lock()
    while True:
        batch_x = batches[i]
        labels_x = next(labels)
        start_y = (855 - crop_length) // 2
        start_x = (1280 - crop_length) // 2
        batch_crops = np.zeros((batch_size, crop_length, crop_length, 3))
        for i in range(batch_crops.shape[0]):
            batch_crops[i] = crop(batch_x[0][i])
        assert(labels_x.shape[0] == batch_crops.shape[0], "labels size = " + str(labels_x.shape[0]))
        yield (batch_crops, labels_x)

        lock.acquire()
        i = i + 1
        if (i >= steps):
            i = 0
        lock.release()
        
# END: generator helper functions

# Creating image generators for each dataset (65/15/20 percent split)
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
Y_train = np_labels[:TRAIN_SIZE]
Y_val = np_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
Y_test = np_labels[TRAIN_SIZE + VAL_SIZE:]
Y_train_gen = label_gen(Y_train, BATCH_SIZE)
Y_val_gen = label_gen(Y_val, BATCH_SIZE)
Y_test_gen = label_gen(Y_test, BATCH_SIZE)

train_crops = crop_generator(train_data_gen, Y_train_gen, 855, BATCH_SIZE, int(6500/BATCH_SIZE))
val_crops = crop_generator(val_data_gen, Y_val_gen, 855, BATCH_SIZE, int(1500/BATCH_SIZE))
test_crops = crop_generator(test_data_gen, Y_test_gen, 855, BATCH_SIZE, int(2000/BATCH_SIZE))

# Define the Keras TensorBoard callback.
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir= f"logs/{time}"
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Defining log reg model
model = tf.keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(10,input_shape=(None, 855 * 855 * 3), activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_data_gen.n//train_data_gen.batch_size
STEP_SIZE_VALID=val_data_gen.n//val_data_gen.batch_size

# fit generator
model.fit_generator(generator=train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_crops,
                    validation_steps= STEP_SIZE_VALID,
                    epochs=10,
                    max_queue_size=10, 
                    callbacks=[tensorboard]
)

STEP_SIZE_TEST=test_data_gen.n//test_data_gen.batch_size

# Re-evaluate the model
loss, acc = model.evaluate_generator(generator = test_crops, steps = STEP_SIZE_TEST, max_queue_size = 10, workers = 1, callbacks = [])
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Restored model, loss: {:5.2f}".format(loss))

model.save_weights('./checkpoints/{time}')
