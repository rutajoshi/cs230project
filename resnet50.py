from dataloader import DataLoader
from models import Models
import constants as c
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

train_crops, val_crops, test_crops = DataLoader().load_data(bins = False)
model = Models().resnet50_v1()

STEP_SIZE_TRAIN=int(6500/c.BATCH_SIZE)
STEP_SIZE_VALID=int(1500/c.BATCH_SIZE)

# Callback functions
# Define the Keras TensorBoard callback.
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('resnet50.h5', monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1)

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

STEP_SIZE_TEST=int(2000/c.BATCH_SIZE)

# Re-evaluate the model
loss, mae = model.evaluate_generator(generator = test_crops, steps = STEP_SIZE_TEST, max_queue_size = 10, workers = 1, callbacks = [])
print("Restored model, loss: {:5.2f}".format(loss))
print("Restored model, mean absolute error: {:5.2f}%".format(mae))
