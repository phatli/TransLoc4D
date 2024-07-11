# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from .layers.pooling_wrapper import PoolingWrapper
from .transformer import LocalFeatureTransformer
from .cbam import SpatialAttGate

from .interpolate_layer import Interpolate


class MinkLoc(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: PoolingWrapper,
        normalize_embeddings: bool = False,
        self_att: bool = False,
        spatial_att: bool = False,
        add_FTU: bool = False,
    ): 
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.linearselfatt = LocalFeatureTransformer() if self_att else None
        # self.linearselfatt = asvt_module() if self_att else None
        self.spatialatt = SpatialAttGate() if spatial_att else None
        self.FTU = Interpolate(64, 256) if add_FTU else None

    def forward(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x, x_conv0 = self.backbone(x)
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )
        if self.FTU is not None:
            x = self.FTU(x, x_conv0)
        if self.linearselfatt is not None:
            x = self.linearselfatt(x, x)
        x = self.pooling(x)
        if hasattr(self.pooling, "stats"):
            self.stats.update(self.pooling.stats)

        # x = x.flatten(1)
        assert (
            x.dim() == 2
        ), f"Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions."
        assert x.shape[1] == self.pooling.output_dim, (
            f"Output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.output_dim}"
        )

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        return {"global": x}

    def print_info(self):
        print("Model class: MinkLoc")
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f"Total parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f"Backbone: {type(self.backbone).__name__} #parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f"Pooling method: {self.pooling.pool_method}   #parameters: {n_params}")
        print("# channels from the backbone: {}".format(self.pooling.in_dim))
        print("# output channels : {}".format(self.pooling.output_dim))
        print(f"Embedding normalization: {self.normalize_embeddings}")

    def forward_local(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)[0].features
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )

        x = F.normalize(x, p=2, dim=1)

        return {"local": x}