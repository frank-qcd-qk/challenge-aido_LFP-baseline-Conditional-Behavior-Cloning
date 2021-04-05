import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from duckieLog.log_util import read_dataset

if __name__ == "__main__":

    # enter your log file location
    log_name = "train"
    log_file = "../{}.log".format(log_name)
    split_size = 0.8

    # enter your output file location
    train_output_file = "../{}_training.tfRecord".format(log_name)
    valid_output_file = "../{}_validation.tfRecord".format(log_name)

    print("Loading dataset... This is gonna take a while...")
    obs, pred, anomaly = read_dataset(
        log_file
    )
    print("Dataset loaded... Converting to TFRecord...")

    train_writer = tf.io.TFRecordWriter(train_output_file)
    valid_writer = tf.io.TFRecordWriter(valid_output_file)

    # Train Test Split
    (
        observation_train,
        observation_valid,
        prediction_train,
        prediction_valid,
        anomaly_train,
        anomaly_valid

    ) = train_test_split(
        obs, pred, anomaly, test_size=1 - split_size, shuffle=False
    )

    for an_observation, an_action, an_anomaly in zip(observation_train, prediction_train, anomaly_train):
        observation_bytes = np.reshape(an_observation, -1).tobytes()
        anomaly_bytes = np.reshape(an_anomaly, -1).tobytes()
        action_bytes = np.reshape(an_action, -1).tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "observation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[observation_bytes])),
            "action": tf.train.Feature(bytes_list=tf.train.BytesList(value=[action_bytes])),
            "anomaly": tf.train.Feature(bytes_list=tf.train.BytesList(value=[anomaly_bytes])),
        }
        ))
        train_writer.write(example.SerializeToString())
    train_writer.close()

    for an_observation, an_action, an_anomaly in zip(observation_valid, prediction_valid, anomaly_valid):
        observation_bytes = np.reshape(an_observation, -1).tobytes()
        anomaly_bytes = np.reshape(an_anomaly, -1).tobytes()
        action_bytes = np.reshape(an_action, -1).tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "observation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[observation_bytes])),
            "action": tf.train.Feature(bytes_list=tf.train.BytesList(value=[action_bytes])),
            "anomaly": tf.train.Feature(bytes_list=tf.train.BytesList(value=[anomaly_bytes])),
        }
        ))
        valid_writer.write(example.SerializeToString())
    valid_writer.close()
