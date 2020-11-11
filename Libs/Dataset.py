import pandas as pd
import os
import torch
import numpy as np
import glob
import torchvision.transforms as transforms
from PIL import Image
import traceback
import copy
import random


class Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 dir,
                 csv_file,
                 label_ids=None,
                 transform=None,
                 eval=False):

        self.dir = dir
        self.data = sorted(
            map(os.path.basename, glob.glob(os.path.join(self.dir, "*.jpg"))))
        self.labels = pd.read_csv(csv_file)
        self.label_ids = label_ids
        self.eval = eval
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.dir, self.data[idx])
        img_name = self.data[idx].split(".jpg")[0]  # filename without jpg
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.eval == False:
            try:
                label = self.labels.loc[self.labels["id"] == int(
                    img_name)]["label"].values[0]
            except IndexError:
                traceback.print_exc()
                print(img_name)
            # train
            return image, self.label_ids[label]
        else:
            # test
            return image, img_name

    def train_test_split(self, train_ratio=0.9):
        test_dataset = copy.deepcopy(self)
        split_point = int(len(self.data) * train_ratio)
        random_index = list(range(len(self.data)))
        random.shuffle(random_index)
        self.data = np.array(self.data)[random_index[:split_point]]
        test_dataset.data = np.array(
            test_dataset.data)[random_index[split_point:]]
        return self, test_dataset


'''
class Data:

    def __init__(self, dir, csv):
        self.filenames = []
        self.labels = None
        self.load_label(csv)
        self.load_dir(dir)

    def __len__(self):
        return len(self.filenames)

    def load_dir(self, dir):
        self.filenames = sorted(os.listdir(dir))
        # print(os.listdir(dir))
        # print(self.filenames)
        return self.filenames

    def load_label(self, file):
        self.labels = pd.read_csv(file)
        # self.labels.sort_values(by=self.labels.columns[0],
        #                         inplace=True)  # sort by id

        return self.labels

    def get_label(self, idx):
        # print(self.filenames[idx])
        # print(self.labels.head())
        print(self.get_filename(idx).split(".jpg")[0])
        return self.labels.loc[self.labels["id"] == int(
            self.get_filename(idx).split(".jpg")[0])]["label"]
        # return self.labels.iloc[int(
        # self.filenames[idx].split(".jpg")[0])]['label']

    def get_filename(self, idx):
        return self.filenames[idx]


class Label:

    def __init__(self):
        self.train = None
        self.test = None
        self.labels = None
'''
