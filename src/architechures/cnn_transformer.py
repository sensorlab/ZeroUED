from torch import nn
import torch
import numpy as np
from typing import Tuple


class TS_CNN_Transformer(nn.Module):

    name = "TS_CNN_Transformer"

    def __init__(
        self,  
        trasnformer,
        cnn, 
        features_size:int=100
    ):
        super(TS_CNN_Transformer, self).__init__()

        self.trasnformer = trasnformer
        self.cnn = cnn

        self.features_size = features_size

        self.combiner = nn.Linear(2 * features_size, features_size)

    def forward(self, x):
        x_cnn = cnn(x)
        x_transformer = transformer(x)

        return self.combiner(
            torch.stack(
                [x_cnn, x_transformer],
                dim = 1
            )
        ), -1

        