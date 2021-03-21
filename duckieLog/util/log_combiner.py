import argparse
import os
import pickle
import sys

from duckieLog import log_util

sys.modules['log_schema'] = log_util

SCHEMA_VERSION = "1.0.0"


class Combiner:
    def __init__(self, output):
        self.log_lists = open("log_list.txt", 'rb')
        self.log_dirs = self.log_lists.read().splitlines()
        self._log_objects = []
        for a_log_file in self.log_dirs:
            if os.path.exists(a_log_file):
                self._log_objects.append(open(a_log_file, 'rb'))
            else:
                print("{} is not a valid log file path! Skipping...".format(a_log_file))
        self._output = open(output, 'wb')
        self.episode_counter = 0
        self.combine(self._log_objects)

    def combine(self, object_list):
        for file_obj in object_list:
            while True:
                try:
                    episode_data = pickle.load(file_obj)
                except EOFError:
                    print("End of log file!")
                    break
                self.commit_episode(episode_data)
                self.episode_counter += 1

        print("Merging Complete. Total Episode: {}".format(self.episode_counter))
        self.close()

    def commit_episode(self, episode):
        pickle.dump(episode, self._output)
        self._output.flush()

    def close(self):
        self._output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="train.log")
    args = parser.parse_args()

    Combiner(args.output)
