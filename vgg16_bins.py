from dataloader import DataLoader
from models import Models

train_crops, val_crops, test_crops = DataLoader().load_data(bins = True)
model = Models().bins_vgg16()

STEP_SIZE_TRAIN=int(6500/BATCH_SIZE)
STEP_SIZE_VALID=int(1500/BATCH_SIZE)

# Callback functions
# Define the Keras TensorBoard callback.
time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/{time}"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('vgg16_best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)
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
print("Restored model, loss: {:5.2f}".format(loss))
print("Restored model, accuracy: {:5.2f}%".format(acc))