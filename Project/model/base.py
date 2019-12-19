import torch


class BaseModel(torch.nn.Module):
    def __init__(self, load_from=None):
        super(BaseModel, self).__init__()
        self.convolutional = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU()
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        if load_from is not None:
            self.load_state_dict(torch.load(load_from))
