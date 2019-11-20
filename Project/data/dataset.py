import os
import pickle
import pandas as pd
from skimage.io import imread
import numpy as np
import torch


label_dict_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'label_dict.pkl')
try:
    with open(label_dict_path, 'rb') as label_dict_file:
        label_dict = pickle.load(label_dict_file)
except:
    label_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_labels.csv')
    label_dict = {row[0]: row[1] for _, row in pd.read_csv(label_file_path).iterrows()}
    with open(label_dict_path, 'wb') as label_dict_file:
        pickle.dump(label_dict, label_dict_file)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, file_names, preview=False):
        self.dir = dir
        self.file_names = file_names
        self.preview = preview

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = imread(os.path.join(self.dir, file_name)).astype(np.float32) / 255
        if not self.preview:
            image = np.moveaxis(image[32:64, 32:64], -1, 0)
        label = label_dict[file_name.replace('.tif', '')]
        return image, label

    def __len__(self):
        return len(self.file_names)

    @property
    def labels(self):
        return np.array([label_dict[file_name.replace('.tif', '')] for file_name in self.file_names])
