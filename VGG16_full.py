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

def crop(img):
    # Note: image_data_format is 'channel_last'
    x = 144
    y = 59
    crop_size = 224
    return img[y:(y+crop_size), x:(x + crop_size), :]

def label_gen(labels, batch_size):
    num = 0
    while True:
        yield labels[num: num + BATCH_SIZE, :]
        num += batch_size
        if (num >= len(labels)):
            num = 0

def crop_gen(batches, labels):
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros((BATCH_SIZE, 224, 224, 3))
        for i in range(BATCH_SIZE):
            batch_crops[i] = crop(batch_x[0][i])
        yield (batch_crops, next(labels))    
        
# END: generator helper functions

# Creating image generators for each dataset (65/15/20 percent split)
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
Y_train = np_labels[:TRAIN_SIZE]
Y_val = np_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
Y_test = np_labels[TRAIN_SIZE + VAL_SIZE:]
Y_train_gen = label_gen(Y_train, BATCH_SIZE)
Y_val_gen = label_gen(Y_val, BATCH_SIZE)
Y_test_gen = label_gen(Y_test, BATCH_SIZE)

train_crops = crop_gen(train_data_gen, Y_train_gen)
val_crops = crop_gen(val_data_gen, Y_val_gen)
test_crops = crop_gen(test_data_gen, Y_test_gen)

# Defining VGG16 model
initial_model = VGG16(weights='imagenet', include_top=False)

for layer in initial_model.layers:
    layer.trainable = False

preds = layers.Flatten()(initial_model.output)
preds.set_shape((None, 25088))
preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)

model = keras.Model(initial_model.input, preds)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN=int(6500/BATCH_SIZE)
STEP_SIZE_VALID=int(1500/BATCH_SIZE)

# Callback functions
# Define the Keras TensorBoard callback.
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('vgg16_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# fit generator
model.fit_generator(generator=train_crops,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_crops,
                    validation_steps= STEP_SIZE_VALID,
                    epochs=10000,
                    verbose=True,
                    use_multiprocessing = False,
                    max_queue_size=10, 
                    callbacks=[es, mc, tensorboard]
)

STEP_SIZE_TEST=int(2000/BATCH_SIZE)

# Re-evaluate the model
loss, acc = model.evaluate_generator(generator = test_crops, steps = STEP_SIZE_TEST, max_queue_size = 10, workers = 1, callbacks = [])
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("Restored model, loss: {:5.2f}".format(loss))
