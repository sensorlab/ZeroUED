from torch import nn
import torch
import numpy as np
from typing import Tuple
from cnn import Simple_CNN_1D
from transformers import TS_Transformer




class TS_CNN_Transformer(nn.Module):

    name = "TS_CNN_Transformer"

    def __init__(
        self,  
        transformer_config,
        simple_cnn_1d_config, 
        features_size,
        n_classes = 10
    ):
        super(TS_CNN_Transformer, self).__init__()

        self.trasnformer = TS_Transformer(transformer_config)
        
        self.cnn = Simple_CNN_1D(simple_cnn_1d_config)

        self.features_size = features_size

        self.combiner = nn.Linear(2 * features_size, features_size)
        self.classif_head = nn.Linear(features_size, n_classes)

    def first_part(self, x):
        self.tmp,_  = self.transformer(x)
        return self.cnn.first_part(x)

    def second_part(self. x):
        x = self.cnn.second_part(x)

        features = self.combiner(torch.stack([x, self.tmp], dim = 1))
        
        return features

    def forward(self, x):
        x_cnn,_ = self.cnn(x)
        
        x_transformer,_ = self.transformer(x)

        features = self.combiner(torch.stack([x_cnn, x_transformer], dim = 1))

        return features, self.classif_head(features)

        