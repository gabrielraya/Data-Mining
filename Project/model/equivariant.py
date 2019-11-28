import torch
from model.standard import StandardModel


class EquivariantModel(torch.nn.Module):
    def __init__(self, load_from=None):
        super(EquivariantModel, self).__init__()
        self.standard = StandardModel()
        if load_from is not None:
            self.load_state_dict(load_from)

    def forward(self, inputs):
        rotoflips = all_flips(inputs) + all_flips(inputs.permute(0, 1, 3, 2))
        results = [self.standard(rotoflip) for rotoflip in rotoflips]
        return torch.stack(results, dim=1).mean(dim=1)


def all_flips(images):
    return [images, torch.flip(images, [-1]), torch.flip(images, [-2]), torch.flip(images, [-1, -2])]
