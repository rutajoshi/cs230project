from dataloader import DataLoader
from models import Models

import constants as c
import pandas as pd
from datetime import datetime
from packaging import version
import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

train_crops, val_crops, test_crops = DataLoader().load_data()
model = Models().vgg16_v3()
model.load_weights("vgg16_v3.h5", by_name = True)

STEP_SIZE_TRAIN=int(6500/c.BATCH_SIZE)
STEP_SIZE_VALID=int(1500/c.BATCH_SIZE)
STEP_SIZE_TEST=int(2000/c.BATCH_SIZE)

# Re-evaluate the model
res = model.evaluate_generator(generator = train_crops, steps = STEP_SIZE_TRAIN, max_queue_size = 10, workers = 1, callbacks = [])
print(res)

# Re-evaluate the model
res = model.evaluate_generator(generator = val_crops, steps = STEP_SIZE_VALID, max_queue_size = 10, workers = 1, callbacks = [])
print(res)
