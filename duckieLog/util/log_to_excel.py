import numpy as np

from duckieLog.log_util import read_dataset

if __name__ == '__main__':
    observation, action, anomaly = read_dataset("../train.log")
    np.savetxt("../action.csv", action, delimiter=",", fmt='%1.3f')
    np.savetxt("../anomaly.csv", anomaly.astype(int), delimiter=",", fmt='% 4d')
