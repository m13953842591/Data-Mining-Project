from model.model import build_from_json
import os
import glob
from data_utils.data_loader import get_generator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import *
from config import *


def find_latest_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        print("checkpoint directory not exist!")
        return None, -1

    paths = glob.glob(os.path.join(checkpoints_dir, ""))
    maxep = -1
    r = None
    for path in paths:
        ep = int(path.split('.')[-1])
        if ep > maxep:
            maxep = ep
            r = path
    return r, maxep


def train(input_dir,
          seq_len,
          epochs=5,
          batch_size=2,
          steps_per_epoch=512,
          val_steps=256,
          model_name="lstm",
          dataset_name="split"):

    config_path = os.path.join("model", model_name + "%d.json" % seq_len)
    if not os.path.exists(config_path):
        raise Exception("invalid model name!")

    model = build_from_json(config_path)

    model.compile(loss="mse", optimizer="adam")
    model.summary()

    train_dir = os.path.join(input_dir, dataset_name, "train")
    test_dir = os.path.join(input_dir, dataset_name, "test")
    
    if not os.path.exists(train_dir):
        raise Exception("train data directory not exist")
    if not os.path.exists(test_dir):
        raise Exception("test data directory not exist")

    train_gen = get_generator(train_dir,
                              seq_len=seq_len,
                              batch_size=batch_size)

    test_gen = get_generator(test_dir,
                             seq_len=seq_len,
                             batch_size=batch_size)

    logdir = os.path.join(WORKSPACE, "logs/%s/%s/%d" % (dataset_name, model_name, seq_len))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    callbacks = [
        TensorBoard(log_dir=logdir),
        ModelCheckpoint("checkpoints\\%s%d.{epoch:02d}-{val_loss:.2f}.hdf5" % (model_name, seq_len),
                        save_best_only=True,
                        monitor='val_loss'),
    ]
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        validation_data=test_gen,
                        validation_steps=val_steps,
                        callbacks=callbacks)

    print("finish training")


if __name__ == '__main__':
    train(input_dir=DATA_PATH,
          seq_len=10,
          epochs=20,
          batch_size=256,
          steps_per_epoch=1190,
          val_steps=490,
          model_name="dnn",
          dataset_name="nodrop")


