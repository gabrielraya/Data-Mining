import torch
from model.base import BaseModel


class EquivariantModel(BaseModel):
    def forward(self, inputs):
        rotoflips = all_rotoflips(inputs)
        convolved = [self.convolutional(rotoflip) for rotoflip in rotoflips]
        rotoflipped_back = undo_rotoflips(convolved)
        mean_pooled = torch.mean(torch.stack(rotoflipped_back, dim=0), dim=0)
        return self.dense(mean_pooled)


def all_rotoflips(images):
    return apply_rotoflips([images] * 8)


def apply_rotoflips(images):
    '''Given a list of image batches, rotate/flip each batch differently'''
    assert len(images) == 8
    return apply_flips(images[:4]) + apply_flips([image.permute(0, 1, 3, 2) for image in images[4:]])


def undo_rotoflips(images):
    assert len(images) == 8
    return apply_flips(images[:4]) + [image.permute(0, 1, 3, 2) for image in apply_flips(images[4:])]


def apply_flips(images):
    assert len(images) == 4
    axes = [[], [-1], [-2], [-1, -2]]
    return [torch.flip(image, axis) for image, axis in zip(images, axes)]
