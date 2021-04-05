import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from duckieModels.cbcNet import cbcNet

MODEL_NAME = "cbcNet_Single_Stage"
logging.basicConfig(level=logging.INFO)

# ! Default Configuration
EPOCHS = 50
INIT_LR = 1e-3
BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
FILE_PREFIX = "train"


class DuckieTrainer:
    def __init__(
            self,
            epochs,
            init_lr,
            batch_size,
            log_name
    ):

        # 0. Setup Folder Structure
        self.create_dirs()
        self.TRAINING_DATASET = "{}_training.tfRecord".format(log_name)
        self.VALIDATION_DATASET = "{}_validation.tfRecord".format(log_name)
        # 1. Load Data
        self.batch_size = batch_size
        self.feature_description = {
            'observation': tf.io.FixedLenFeature([], tf.string),
            'action': tf.io.FixedLenFeature([], tf.string),
            'anomaly': tf.io.FixedLenFeature([], tf.string),
        }
        raw_train_dataset = tf.data.TFRecordDataset(self.TRAINING_DATASET)
        raw_valid_dataset = tf.data.TFRecordDataset(self.VALIDATION_DATASET)

        # 2. preprocess dataset
        self.train_dataset = raw_train_dataset.map(self.parse_functions).map(self.ingest_tfRecord).batch(
            self.batch_size)
        self.valid_dataset = raw_valid_dataset.map(self.parse_functions).map(self.ingest_tfRecord).batch(
            self.batch_size)

        # 2. Setup Model
        model = cbcNet.get_model(init_lr, epochs)
        callbacks_list = self.configure_callbacks()

        # 11. GO!
        history = model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=batch_size,
            verbose=0,
        )

        model.save(f"trainedModel/{MODEL_NAME}.h5")

    def create_dirs(self):
        try:
            dirname, _ = os.path.split(os.path.abspath(__file__))
            Path(os.path.join(dirname, "trainedModel")).mkdir(parents=True, exist_ok=True)
        except OSError:
            print(
                "Create folder for trained model failed. Please check system permissions."
            )
            exit()

    def parse_functions(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.feature_description)

    def ingest_tfRecord(self, tf_example):
        """Given a tf_example dict, separates into feature_dict and target_dict"""
        # print(tf_example)
        raw_img = tf_example['observation']
        raw_anomaly = tf_example['anomaly']
        raw_pred = tf_example['action']
        observation = tf.reshape(tf.io.decode_raw(raw_img, tf.uint8), shape=(150, 200, 3))
        anomaly = tf.io.decode_raw(raw_anomaly, tf.int32)
        action = tf.io.decode_raw(raw_pred, tf.float64)
        return (observation, (action, anomaly))

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(
                f'{MODEL_NAME}-{datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}'
            )
        )

        filepath1 = f"trainedModel/{MODEL_NAME}_Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}_Best_Loss.h5"
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
            filepath2, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )

        return [checkpoint1, checkpoint2, tensorboard]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")

    parser.add_argument(
        "--epochs", help="Set the total training epochs", default=EPOCHS
    )
    parser.add_argument(
        "--learning_rate", help="Set the initial learning rate", default=INIT_LR
    )
    parser.add_argument(
        "--batch_size", help="Set the batch size", default=BATCH_SIZE)
    parser.add_argument(
        "--training_dataset", help="Set the training TF Record prefix name", default=FILE_PREFIX
    )

    args = parser.parse_args()
    DuckieTrainer(
        epochs=int(args.epochs),
        init_lr=float(args.learning_rate),
        batch_size=int(args.batch_size),
        log_name=args.training_dataset,
    )
