from tensorflow._api.v1 import keras
from models import *
from config import *
from generator import BatchGenerator
import os

train_data_path = DATA_PATH + "\\train"
test_data_path = DATA_PATH + "\\test"
model_save_path = os.path.join(MODEL_PATH, 'lstm_1.h5')
model = DNN

model.compile(optimizer=keras.optimizers.Adam(lr=LR, decay=DECAY),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_absolute_error])

print(model.summary())

total_train_sample = len(os.listdir(train_data_path)) * FILE_SIZE
total_test_sample = len(os.listdir(test_data_path)) * FILE_SIZE

train_generator = BatchGenerator(train_data_path, TIME_STEP, BATCH_SIZE,
                                 x_units=NUM_FEATURE, y_units=NUM_LABEL,
                                 skip_step=SKIP_SIZE)
test_generator = BatchGenerator(test_data_path,  TIME_STEP, BATCH_SIZE,
                                x_units=NUM_FEATURE, y_units=NUM_LABEL,
                                skip_step=SKIP_SIZE)

history = model.fit_generator(
    generator=train_generator.generate(),
    steps_per_epoch=train_generator.steps_per_epoch,
    validation_data=test_generator.generate(),
    validation_steps=test_generator.steps_per_epoch,
    epochs=NUM_EPOCH,
    callbacks=[keras.callbacks.TensorBoard(log_dir=LOGDIR)],
)

model.save(model_save_path)
