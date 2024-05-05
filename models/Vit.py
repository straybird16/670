import torch
import torchvision
import torch.nn as nn

from collections import OrderedDict
from functools import partial
from typing import Callable, List
from torch.nn.modules import LayerNorm, Module
from torchvision.models.vision_transformer import ConvStemConfig

class ViT(nn.Module):

    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, decoder_dim:int=32, dropout: float = 0, attention_dropout: float = 0, num_classes: int = 1000, representation_size: int | None = None, norm_layer: Callable[..., Module] =partial(nn.LayerNorm, eps=1e-6), conv_stem_configs: List[ConvStemConfig] | None = None):
        super().__init__()
        # attributes
        self.image_size, self.patch_size = image_size, patch_size
        self.grid_num = (image_size//patch_size)**2
        decoder_dim=image_size  # output the same size as input image size for now
        self.decoder_dim = decoder_dim
        # base transformer
        self.transformer = torchvision.models.VisionTransformer(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size, norm_layer, conv_stem_configs)
        # Decoder block for feature maps
        self.decoder = DecoderBlock(input_dim=self.grid_num, output_dim=decoder_dim)

        self.conv1x1Block1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=64, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh(),
            nn.BatchNorm2d(num_features=64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Tanh(),
        )
        self.conv1x1Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        return self.transformer(x)
    
    def reconstruct(self, x):
        # In contrary to prediction, we discard the class token and get feature maps from the rest
        x = self.transformer._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.transformer.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.transformer.encoder(x)
        # discard the output token
        clt = x[:,0]
        clt = clt.view(clt.shape + (1, 1))
        x = x[:, 1:].permute(0, 2, 1)  # (N, grid_num, hidden_dim) -> (N, hidden_dim, grid_num)
        x = self.decoder(x)  #  (N, hidden_dim, decoder_dim*decoder_dim)
        x = x * clt  # leverage classification knowledge to reconstruct from __hidden_dim__ channels of features
        
        x = self.conv1x1Block1(x)
        x = self.conv1x1Block2(x)  
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> nn.Module:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        output_dim = output_dim * output_dim
        
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(p=0.1),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x:torch.Tensor):
        # x is of shape (N x L x input_dim)
        # first transform encodings to shape (N x L x output_dim*output_dim)
        x = self.decoder(x)
        # then de-flatten the input to shape (N x C x output_dim, output_dim)
        x = x.view(x.shape[0],-1,self.output_dim, self.output_dim)
        return x