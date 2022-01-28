import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import (
    ColorJitter,
    RandomChannelShuffle,
    RandomHorizontalFlip,
    RandomThinPlateSpline,
)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transform1 = K.augmentation.RandomHorizontalFlip(p=0.6)
        self.transform2 = K.augmentation.RandomVerticalFlip(p=0.6)
        self.transform3 = K.augmentation.RandomCrop(200)
        self.transform4 = K.augmentation.RandomRotation(90)
        self.transform5 = K.augmentation.CenterCrop(200)
        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        y = x
        x_out = self.transform1(x)  # BxCxHxW
        y_out = torch.utils.data.ConcatDataset([x, x_out])
        # x_out  = self.transform2(x)
        # y_out =torch.utils.data.ConcatDataset([y_out,x_out])
        x_out = self.transform3(x)
        y_out = torch.utils.data.ConcatDataset([y_out, x_out])
        x_out = self.transform4(x)
        y_out = torch.utils.data.ConcatDataset([y_out, x_out])
        x_out = self.transform5(x)
        y_out = torch.utils.data.ConcatDataset([y_out, x_out])
        if self._apply_color_jitter:
            y_out = self.jitter(y_out)
        return y_out