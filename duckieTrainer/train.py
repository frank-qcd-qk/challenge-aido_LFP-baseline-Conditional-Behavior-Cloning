import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from duckieLog.log_util import read_dataset
from duckieModels.cbcNet import cbcNet

MODEL_NAME = "cbcNet"
logging.basicConfig(level=logging.INFO)

# ! Default Configuration
EPOCHS = 50
INIT_LR = 1e-3
BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
TRAINING_DATASET = "train.log"


class DuckieTrainer:
    def __init__(
            self,
            epochs,
            init_lr,
            batch_size,
            log_file,
            split,
    ):

        # 0. Setup Folder Structure
        self.create_dirs()

        # 1. Load Data
        self.observation, self.prediction, self.anomaly = read_dataset(
            log_file
        )

        # 2. Split training and testing
        (
            observation_train,
            observation_valid,
            prediction_train,
            prediction_valid,
            anomaly_train,
            anomaly_valid

        ) = train_test_split(
            self.observation, self.prediction, self.anomaly, test_size=1 - split, shuffle=False
        )
        print(np.sum(anomaly_train))
        model = cbcNet.get_model(init_lr, epochs)
        callbacks_list = self.configure_callbacks()

        # 11. GO!
        history = model.fit(
            x=observation_train,
            y={"tf.where": prediction_train, "Anomaly_Out": anomaly_train},
            validation_data=(
                observation_valid,
                {"tf.where": prediction_valid, "Anomaly_Out": anomaly_valid},
            ),
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

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(
                f'{MODEL_NAME}-{datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}'
            )
        )

        filepath1 = f"trainedModel/{MODEL_NAME}Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}Best_Loss.h5"
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
