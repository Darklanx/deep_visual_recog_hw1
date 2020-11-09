import pandas as pd
import os


class Data:

    def __init__(self):
        self.train = None
        self.test = None
        self.labels = None

    def load_dir(self, dir):
        return os.listdir(dir)[0:15]

    def load_label(self, file):
        self.labels = pd.read_csv(file)
        return self.labels

    def get_label(self, idx):
        return self.labels.iloc[idx]['label']