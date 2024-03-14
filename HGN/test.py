import torch.nn as nn
import torch

device='cuda'
layer =nn.Conv2d(in_channels=1,
                  out_channels=10,
                  kernel_size=3,
                  stride=1,
                  padding=1).to(device)

rand_image_tensor = torch.randn(size=(1, 1, 28, 28)).to(device)

print(layer(rand_image_tensor).shape)

