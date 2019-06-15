from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import json
from data_utils.data_loader import DataLoader
from global_var import *
from model.model import build_model
import os
import six
import glob


def find_latest_checkpoint(checkpoints_path):
    paths = glob.glob(checkpoints_path + ".*")
    maxep = -1
    r = None
    for path in paths:
        ep = int(path.split('.')[-1])
        if ep > maxep:
            maxep = ep
            r = path
    return r, maxep


def train(input_dir,
          configs,
          future_n,
          split,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          val_data=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          steps_per_epoch=512,
          callbacks=None):

    model = build_model(configs=configs)

    model.summary()

    if auto_resume_checkpoint and (not checkpoints_path is None):
            model.load_weights(checkpoints_path)

    data_loader = DataLoader(input_dir, split=split)

    train_gen = DataLoader.generate_train_batch()

    if not validate:
        for ep in range(latest_ep + 1, latest_ep + 1 + epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen,
                                steps_per_epoch,
                                epochs=1,
                                callbacks=callbacks)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".models." + str(ep))
            print("Finished Epoch", ep)
    else:
        for ep in range(latest_ep + 1, latest_ep + 1 + epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen,
                                steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,
                                callbacks=callbacks,
                                epochs=1)

            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".models." + str(ep))
            print("Finished Epoch", ep)


if __name__ == '__main__':
    train_images_dir = DATA_PATH + "/train_images"
    val_images_dir = DATA_PATH + "/val_images"
    train_labels_dir = DATA_PATH + "/train_labels"
    val_labels_dir = DATA_PATH + "/val_labels"
    tensorboard = TensorBoard(log_dir='./logs/%s/' % MODEL_NAME)
    train(model=MODEL_NAME,
          train_images=train_images_dir,
          train_annotations=train_labels_dir,
          input_height=INPUT_HEIGHT,
          input_width=INPUT_WIDTH,
          n_classes=NUM_CLASS,
          verify_dataset=False,
          checkpoints_path="checkpoints/" + MODEL_NAME,    # don't add '/' in the end
          epochs=EPOCH,
          batch_size=BATCH_SIZE,
          validate=True,
          val_images=val_images_dir,
          val_annotations=val_labels_dir,
          val_batch_size=BATCH_SIZE,
          auto_resume_checkpoint=AUTO_RESUME,
          load_weights=None,
          steps_per_epoch=STEPS_PER_EPOCH,
          optimizer_name=OPTIMIZER,
          callbacks=[tensorboard]
          )

