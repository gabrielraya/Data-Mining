import torch


def StandardModel(load_from=None):
    result = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(384, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )
    if load_from is not None:
        result.load_state_dict(torch.load(load_from))
    return result
