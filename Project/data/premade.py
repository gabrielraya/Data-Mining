import os
import torch
from data.dataset import Dataset


image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train')
all_file_names = sorted(os.listdir(image_dir))
train_file_names = all_file_names[:-256]
valid_file_names = all_file_names[len(train_file_names):]

preview = Dataset(image_dir, all_file_names[:4], preview=True)
train = Dataset(image_dir, train_file_names)
valid = Dataset(image_dir, valid_file_names)

train_loader = torch.utils.data.DataLoader(train, 32, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid, 32, shuffle=False, num_workers=2)
