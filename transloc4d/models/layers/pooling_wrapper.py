import torch.nn as nn
import MinkowskiEngine as ME
from .TransVLAD3d import TransVLAD3d, find_initcache
from .pooling import MAC, SPoC, GeM, NetVLADWrapper, GeM_tensor

class PoolingWrapper(nn.Module):
    def __init__(self, pool_method, in_dim, output_dim, vlad_init=True, num_clusters=8):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim

        if pool_method == 'MAC':
            # Global max pooling
            assert in_dim == output_dim
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            # Global average pooling
            assert in_dim == output_dim
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            # Generalized mean pooling
            assert in_dim == output_dim
            self.pooling = GeM(input_dim=in_dim)
        elif pool_method == 'GeM_tensor':
            # Generalized mean pooling
            assert in_dim == output_dim
            self.pooling = GeM_tensor(input_dim=in_dim)
        elif self.pool_method == 'netvlad':
            # NetVLAD
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=False)
        elif self.pool_method == 'netvladgc':
            # NetVLAD with Gating Context
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=True)
        elif self.pool_method.lower() == 'transvlad3d':
            # NetVLAD with Gating Context
            self.pooling = TransVLAD3d(dim=in_dim, num_clusters=num_clusters, ghost_weighting=True, param_norm=True, num_ghost=9)
            self.output_dim = output_dim * num_clusters
            # find inicache, if not, create clusters and init
            if vlad_init:
                find_initcache(self.pooling, num_clusters=num_clusters)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x: ME.SparseTensor):
        return self.pooling(x)
