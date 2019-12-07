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

import constants as c

# tensorflow stuff
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import backend as K

class Models():
    
    def __load_vgg16__(self):
        initial_model = VGG16(weights='imagenet', include_top=False)

        for layer in initial_model.layers:
            layer.trainable = False
            
        return initial_model
    
        # custom loss function
    def bins_mean_squared_error(self):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            yt = tf.argmax(y_true, axis=-1) * 10**(-c.N_BINS)
            yp = tf.argmax(y_pred, axis=-1) * 10**(-c.N_BINS)
            return tf.keras.losses.MSE(yt, yp)  
        # Return a function
        return loss
    
    # vgg16 with transfer learning
    def vgg16_v1(self):
        initial_model = self.__load_vgg16__()
        preds = layers.Flatten()(initial_model.output)
        preds.set_shape((None, 25088))
        preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 25088), trainable=True)(preds)

        model = keras.Model(initial_model.input, preds)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
    
    # Version 2 - this passes the output into:
    # a conv layer, tanh activation, spatial dropout, flattens, and into a dense layer
    def vgg16_v2(self):
        initial_model = self.__load_vgg16__()
        preds = layers.Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
        preds = layers.Activation('tanh')(preds)
        preds = layers.SpatialDropout2D(0.4)(preds)
        preds = layers.Flatten()(preds)
        preds.set_shape((None, 1470))
        preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 1470), trainable=True)(preds)
        
        model = keras.Model(initial_model.input, preds)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
        
    # Version 3 - this passes the output into:
    # a conv layer, tanh activation, another conv layer, spatial dropout, flattens, and into a dense layer
    def vgg16_v3(self):
        initial_model = self.__load_vgg16__()
        preds = layers.Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
        preds = layers.Activation('tanh')(preds)
        preds = layers.Conv2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last') (preds)
        preds = layers.SpatialDropout2D(0.4)(preds)
        preds = layers.Flatten()(preds)
        preds.set_shape((None, 1296))
        preds = layers.Dense(10, activation='sigmoid', input_shape=(None, 1296), trainable=True)(preds)
        
        model = keras.Model(initial_model.input, preds)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
        
    # bins 
    def bins_vgg16(self, opt = 1, loss = 0):
        initial_model = self.__load_vgg16__()
        
        if (opt == 1):
            preds = layers.Flatten()(initial_model.output)
            preds.set_shape((None, 25088))

            x1 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x2 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x3 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x4 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x5 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x6 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x7 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x8 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x9 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            x10 = layers.Dense(c.N_BINS, activation='softmax', input_shape=(None, 25088), trainable=True)(preds)
            preds = layers.Concatenate(axis = 1)([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
            
#         elif (opt == 2):
#             preds = Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
#             preds = Activation('tanh')(preds)
#             preds = SpatialDropout2D(0.4)(preds)
#             preds = Flatten()(preds)
#             preds.set_shape((None, 1470))
#         else:
#             preds = Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (initial_model.output)
#             preds = Activation('tanh')(preds)
#             preds = Conv2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last') (preds)
#             preds = SpatialDropout2D(0.4)(preds)
#             preds = Flatten()(preds)
#             preds.set_shape((None, 1296))
        model = keras.Model(initial_model.input, preds)
        # Compile the model
        
        if (loss == 1):
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.compile(loss=self.bins_mean_squared_error(), optimizer='adam', metrics=['accuracy'])

        return model
           
    # Added layers equivalent to vgg_v1
    def resnet50_v1(self):
        # Get the pretrained ResNet50 model. We set include_top to False to specify that we don't want the classification layers
        initial_model = ResNet50(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(224,224,3)))
 
        # Set all existing ResNet layers to be untrainable (freeze weights)
        for layer in initial_model.layers:
            layer.trainable = False
 
        preds = initial_model.output
        preds = layers.Flatten()(preds)
        preds.set_shape((None, 100352))
        preds = layers.Dense(10, activation='sigmoid', trainable=True)(preds)
 
        model = keras.Model(initial_model.input, preds)
        print(model.summary())
        opt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model

 
    # NOTE: fixed added layers
    # Added layers equivalent to vgg_v2
    def resnet50_v2(self):
        # Get the pretrained ResNet50 model. We set include_top to False to specify that we don't want the classification layers
        initial_model = ResNet50(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(224,224,3)))
        
        # Set all existing ResNet layers to be untrainable (freeze weights)
        for layer in initial_model.layers:
            layer.trainable = False

        preds = initial_model.output
        preds = layers.Conv2D(30, 5, strides=(1, 1), padding='valid', data_format='channels_last') (preds)
        preds = layers.Activation('tanh')(preds)
        preds = layers.SpatialDropout2D(0.4)(preds)
        preds = layers.Flatten()(preds)
        preds.set_shape((None, 270))
        preds = layers.Dense(10, activation='sigmoid', trainable=True)(preds)

        model = keras.Model(initial_model.input, preds)
        opt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model 
