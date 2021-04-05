from duckieLog.log_util import read_dataset, Logger, Step

if __name__ == '__main__':
    datagen = Logger(None, log_file="../cherripicked.log")
    observation, action, anomaly = read_dataset("../train.log")
    for an_observation, an_action, an_anomaly in zip(observation, action, anomaly):
        if an_anomaly:
            step = Step(an_observation, None, an_action, False, 0)
            datagen.log(step, None)
    datagen.on_episode_done()
    datagen.close()
