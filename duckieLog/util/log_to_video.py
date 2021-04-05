import cv2
import imageio

from duckieLog.log_util import read_dataset

if __name__ == '__main__':
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
