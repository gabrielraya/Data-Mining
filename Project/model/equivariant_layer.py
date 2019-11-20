import torch


class EquivariantLayer(torch.nn.Module):
    def __init__(self, out_channels, *args, **kwargs):
        assert out_channels % 8 == 0
        super(EquivariantLayer, self).__init__()
        self.base_conv = torch.nn.Conv2d(*args, out_channels=out_channels // 8, **kwargs)

    def forward(self, inputs):
        all_transformations = all_flips(inputs) + all_flips(inputs.permute(0, 1, 3, 2))
        all_convolved = [self.base_conv(transformation) for transformation in all_transformations]
        return torch.cat(all_convolved, dim=1)


def all_flips(images):
    return [images, torch.flip(images, [-1]), torch.flip(images, [-2]), torch.flip(images, [-1, -2])]
