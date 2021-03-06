import cv2
import imageio
import numpy as np

from duckieLog.log_util import read_dataset, Logger, Step


def filter_anomaly_data():
    datagen = Logger(None, log_file="../cherripicked.log")
    observation, action, anomaly = read_dataset("../train.log")
    for an_observation, an_action, an_anomaly in zip(observation, action, anomaly):
        if an_anomaly:
            step = Step(an_observation, None, an_action, False, 0)
            datagen.log(step, None)
    datagen.on_episode_done()
    datagen.close()
    return


def log_to_excel():
    observation, action, anomaly = read_dataset("../train.log")
    np.savetxt("../action.csv", action, delimiter=",", fmt='%1.3f')
    np.savetxt("../anomaly.csv", anomaly.astype(int), delimiter=",", fmt='% 4d')
    return


def log_to_video():
    observation, action, anomaly = read_dataset("/home/frank/duckietown/aido6/log_sources/Mar8_zig_zag_more_ducks.log")
    height, width, layers = observation[0].shape
    print(observation[0].shape)
    writer = imageio.get_writer("../TrainingDataVisualization.mp4", format='mp4', mode='I', fps=60.0)
    for an_observation, an_action, an_anomaly in zip(observation, action, anomaly):
        canvas = cv2.cvtColor(an_observation, cv2.COLOR_YUV2RGB)
        x = an_action[0]
        z = an_action[1]
        cv2.rectangle(canvas, (20, 100), (30, int(100 - 90 * x)),
                      (76, 84, 255), cv2.FILLED)
        cv2.rectangle(canvas, (100, 130), (int(100 - 90 * z), 140), (76, 84, 255), cv2.FILLED)
        if an_anomaly:
            cv2.rectangle(canvas, (170, 0), (200, 30), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(canvas, (170, 0), (200, 30), (0, 255, 0), cv2.FILLED)
        cv2.imshow('Playback', canvas)
        cv2.waitKey(20)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.close()


if __name__ == '__main__':
    filter_anomaly_data()
