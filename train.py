from model.model import build_model
import os
import glob
from data_utils.data_loader import get_generator
from keras.callbacks import TensorBoard, ModelCheckpoint
import json


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
          test_batch_size=2,
          steps_per_epoch=512,
          model_name="lstm"):

    config_path = os.path.join("model", model_name + ".json")
    if not os.path.exists(config_path):
        raise Exception("invalid model name!")

    with open(config_path, 'r') as f:
        configs = json.load(f)

    model = build_model(configs)

    model.summary()

    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")
    
    if not os.path.exists(train_dir):
        raise Exception("train data directory not exist")
    if not os.path.exists(test_dir):
        raise Exception("test data directory not exist")

    train_gen = get_generator(train_dir,
                              seq_len=seq_len,
                              batch_size=batch_size)

    test_gen = get_generator(test_dir,
                             seq_len=seq_len,
                             batch_size=test_batch_size)

    callbacks = [
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint("checkpoints\\%s.{epoch:02d}-{val_loss:.2f}.hdf5" % model_name,
                        save_best_only=True,
                        monitor='val_loss'),
    ]
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_gen,
                        validation_steps=100,
                        callbacks=callbacks)

    print("finish training")


if __name__ == '__main__':
    pass


