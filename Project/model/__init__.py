import torch
from model.builder import build_model
from model.equivariant_layer import EquivariantLayer

standard = build_model(conv_layer_type=torch.nn.Conv2d)
equivariant = build_model(conv_layer_type=EquivariantLayer)
