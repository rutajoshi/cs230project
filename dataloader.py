import math
import numpy as np
import pandas as pd
import tensorflow as tf
import constants as c
from tensorflow import keras

class DataLoader:
    
    # private 
    def __init__(self):
        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    def __get_labels__(self):
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
        return vector_value.values
    
    def __crop__(self, img):
        return img[c.Y_TOP:(c.Y_TOP+c.CROP_SIZE), c.X_LEFT:(c.X_LEFT + c.CROP_SIZE), :]

    def __label_gen__(self, labels, bins = False):
        num = 0
        while True:
            if (bins):
                l = []
                for i in range(10):
                    l.append(labels[i][num : num + c.BATCH_SIZE])
                yield l
            else:
                yield labels[num: num + c.BATCH_SIZE, :]
            num += c.BATCH_SIZE
            if (num >= len(labels)):
                num = 0

    def __crop_gen__(self, batches, labels):
        while True:
            batch_x = next(batches)
            batch_crops = np.zeros((c.BATCH_SIZE, 224, 224, 3))
            for i in range(c.BATCH_SIZE):
                batch_crops[i] = self.__crop__(batch_x[0][i])
            yield (batch_crops, next(labels))  
            
    def __create_gen__(self, path, labels, bins = False):
        data_gen = self.image_generator.flow_from_directory(directory=str(path),
                                                     target_size=(int(c.IMG_HEIGHT * .4), int(c.IMG_WIDTH * .4)),
                                                     batch_size=c.BATCH_SIZE,
                                                     shuffle = False,
                                                     classes = None)
        Y_gen = self.__label_gen__(labels, bins)
        return self.__crop_gen__(data_gen, Y_gen)
    
    # bins one hot preprocess
    def __convert_to_indices__(self, arr, binsPower):
        arr = np.around(arr, decimals=binsPower)
        arr = np.multiply(arr, 10**(binsPower))
        return arr.astype(int)

    def __onehot_initialization__(self, a, binsPower):
        ncols = 10 **(binsPower) + 1
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self.__all_idx__(a, axis=2)] = 1
        return out

    def __all_idx__(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def __convert_onehot__(self, arr, binPower):
        # split into 10 and then convert to indices
        l = []
        for i in range(10):   
            tmp = self.__convert_to_indices__(arr[:,i], binPower)
            tmp = self.__onehot_initialization__(tmp, binPower)
            l.append(tmp)
        return l
    
    # public
    def load_data(self, bins = False):
        labels = self.__get_labels__()
        if (bins):
            labels = self.__convert_onehot__(labels, c.BIN_POWER)
        train_gen = self.__create_gen__("./data_split/train_data", labels[:c.TRAIN_SIZE], bins)
        val_gen = self.__create_gen__("./data_split/val_data", labels[c.TRAIN_SIZE:c.TRAIN_SIZE + c.VAL_SIZE], bins)
        test_gen = self.__create_gen__("./data_split/test_data", labels[c.TRAIN_SIZE + c.VAL_SIZE:], bins)
        return train_gen, val_gen, test_gen
    
    