#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16

from datetime import datetime
from packaging import version

# constants
BATCH_SIZE = 25

IMG_HEIGHT = 224
IMG_WIDTH = 224
DATASET_SIZE = 10000
TRAIN_SIZE = 6500
VAL_SIZE = 1500
TEST_SIZE = int(DATASET_SIZE * .2)

STEPS_PER_EPOCH = np.ceil(TRAIN_SIZE/BATCH_SIZE)





# In[10]:


labels = pd.read_csv('../data.csv', float_precision='road_trip')
labels[['base color R','base color G', 'base color B']] = labels['base color'].str.split(",",expand=True) 
labels[['specular color R','specular color G', 'specular color B']] = labels['specular color'].str.split(",",expand=True) 
labels.drop(columns =["base color"], inplace = True) 
labels.drop(columns =["specular color"], inplace = True) 
labels["base color R"] = labels["base color R"].str[1:]
labels["base color B"] = labels["base color B"].str[:-1]
labels["specular color R"] = labels["specular color R"].str[1:]
labels["specular color B"] = labels["specular color B"].str[:-1]
labels["img_name"] = 'images/'+labels["img_name"].str[:-3]+"jpg"
labels2 = pd.read_csv('../big_dataset.csv', float_precision='road_trip')
labels2["img_name"] = 'images/'+labels2["img_name"].str[:-3]+"jpg"
labels = labels.append(labels2)
cols = ['img_name', 'specular roughness', 'specular', 'metalness','base color R','base color G', 'base color B','specular color R',     'specular color G','specular color B'] 
labels = labels[cols]
vector_values = labels.iloc[:,1:]
vector_values = vector_values.astype(float)
np_labels = vector_values.values


# In[6]:


filelist = ["../cropped_images/" + str(i) + ".jpg" for i in range(TRAIN_SIZE)]
trainImages = np.array([np.array(imageio.imread(fname)) for fname in filelist])

filelist = ["../cropped_images/" + str(i) + ".jpg" for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE)]
valImages = np.array([np.array(imageio.imread(fname)) for fname in filelist])

filelist = ["../cropped_images/" + str(i) + ".jpg" for i in range(TRAIN_SIZE+VAL_SIZE,TRAIN_SIZE+VAL_SIZE+TEST_SIZE)]
testImages = np.array([np.array(imageio.imread(fname)) for fname in filelist])


# In[16]:


X_train = trainImages
X_val = valImages
X_test = testImages


# In[19]:


# This is the keras image generator that uses the above 2 functions to generate training/test batches
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Creating label generators for each dataset (65/15/20 percent split)
# (Set up the label datasets to be passed into the generators)
Y_train = np_labels[:TRAIN_SIZE]
Y_val = np_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
Y_test = np_labels[TRAIN_SIZE + VAL_SIZE:]


# Define the Keras TensorBoard callback
# Also make log directories based on timestamp
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Get the pretrained VGG model. We set include_top to False to specify that we don't want the classification layers
initial_model = VGG16(weights='imagenet', include_top=False)
# Set all existing VGG16 layers to be untrainable (freeze weights)
for layer in initial_model.layers:
    layer.trainable = False

# ARCHITECTURE CHANGES ARE HERE

# Version 1 - this flattens the output and passes into a dense 10 layer output
preds = layers.Flatten()(initial_model.output)
preds.set_shape((None, 25088))
preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)


# Initialize and compile keras model
model = keras.Model(initial_model.input, preds)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Set the train and validation step sizes by calculating the number of
# batches in the train and test sets respectively
STEP_SIZE_TRAIN=int(6500/BATCH_SIZE)
STEP_SIZE_VALID=int(1500/BATCH_SIZE)

# Fit the model to the training data and validate on the validation data
model.fit_generator(image_generator.flow(X_train, Y_train, BATCH_SIZE),
                    steps_per_epoch=STEP_SIZE_TRAIN, 
                    epochs=10, 
                    validation_data = image_generator.flow(X_val, Y_val, BATCH_SIZE),
                    validation_steps= STEP_SIZE_VALID,
                    callbacks=[tensorboard])

STEP_SIZE_TEST=int(2000/BATCH_SIZE)

# Re-evaluate the model on the test data
loss, mae = model.evaluate_generator(image_generator.flow(X_test, Y_test, BATCH_SIZE), steps = STEP_SIZE_TEST, callbacks = [])
print("Restored model, mean absolute error: {:5.2f}%".format(mae))
print("Restored model, loss: {:5.2f}".format(loss))

# Save the weights in a checkpoint
model.save_weights('./checkpoints/{time}')


# In[ ]:




