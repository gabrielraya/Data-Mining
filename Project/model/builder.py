import torch


def build_model(conv_layer_type):
    return torch.nn.Sequential(
        conv_layer_type(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        conv_layer_type(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        conv_layer_type(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2)
    )
