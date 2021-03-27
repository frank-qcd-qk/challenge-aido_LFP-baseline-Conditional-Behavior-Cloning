import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from duckieLog.log_util import read_dataset
from duckieModels.cbcNetv2 import cbcNetv2

MODEL_NAME = "cbcNet"
logging.basicConfig(level=logging.INFO)

# ! Default Configuration
EPOCHS = 100
INIT_LR = 1e-3
BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
TRAINING_DATASET = "train.log"
OPMODE = "Anomaly"
OPMODE = "BC"


class DuckieTrainer:
    def __init__(
            self,
            epochs,
            init_lr,
            batch_size,
            log_file,
            split,
    ):
        self.batch_size = batch_size
        # 0. Setup Folder Structure
        self.create_dirs()

        # 1. Load Data
        self.observation, self.prediction, self.anomaly = read_dataset(
            log_file
        )

        # 2. Split training and testing
        (
            self.observation_train,
            self.observation_valid,
            self.prediction_train,
            self.prediction_valid,
            self.anomaly_train,
            self.anomaly_valid

        ) = train_test_split(
            self.observation, self.prediction, self.anomaly, test_size=1 - split, shuffle=False
        )
        self.cmd_model, self.anomaly_model = cbcNetv2.get_model(init_lr, epochs)
        if OPMODE == "Anomaly":
            self.train_anomaly_detector()
        else:
            self.train_bc_net()

    def train_anomaly_detector(self):
        callbacks_list = self.configure_callbacks("Anomaly")

        # 11. GO!
        history = self.anomaly_model.fit(
            x=self.observation_train,
            y=self.anomaly_train,
            validation_data=(
                self.observation_valid,
                self.anomaly_valid,
            ),
            epochs=EPOCHS,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

    def train_bc_net(self):
        callbacks_list = self.configure_callbacks("BC")

        # 11. GO!
        history = self.cmd_model.fit(
            x=[self.observation_train, self.anomaly_train],
            y=self.prediction_train,
            validation_data=(
                [self.observation_valid, self.anomaly_valid],
                self.prediction_valid
            ),
            epochs=EPOCHS * 2,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

    def create_dirs(self):
        try:
            dirname, _ = os.path.split(os.path.abspath(__file__))
            Path(os.path.join(dirname, "trainedModel")).mkdir(parents=True, exist_ok=True)
        except OSError:
            print(
                "Create folder for trained model failed. Please check system permissions."
            )
            exit()

    def configure_callbacks(self, type):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(
                f'{MODEL_NAME}-{type}-{datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}'
            )
        )

        filepath1 = f"trainedModel/{MODEL_NAME}-{type}-Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}-{type}-Best_Loss.h5"
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
        "--training_dataset", help="Set the training log file name", default=TRAINING_DATASET
    )
    parser.add_argument(
        "--split", help="Set the training and test split point (input the percentage of training)",
        default=TRAIN_PERCENT
    )

    args = parser.parse_args()

    DuckieTrainer(
        epochs=int(args.epochs),
        init_lr=float(args.learning_rate),
        batch_size=int(args.batch_size),
        log_file=args.training_dataset,
        split=float(args.split)
    )
