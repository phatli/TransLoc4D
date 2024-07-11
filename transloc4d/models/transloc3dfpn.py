import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np

from .transloc3dtransformer import TransformerBlock


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_sparse = self.avg_pool(x)

        # Apply 1D convolution along the channel dimension
        y = self.conv(y_sparse.F.unsqueeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).squeeze(-1)
        # y is (batch_size, channels) tensor

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor

        y_sparse = ME.SparseTensor(y, coordinate_manager=y_sparse.coordinate_manager,
                                   coordinate_map_key=y_sparse.coordinate_map_key)
        # y must be features reduced to the origin
        return self.broadcast_mul(x, y_sparse)


class SelectiveInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # receptive field 1x1
        if in_channels != out_channels:
            self.conv1x1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dimension=3),
                ME.MinkowskiBatchNorm(
                    out_channels)
            )
        else:
            self.conv1x1 = lambda x: x

        # receptive field 3x3
        self.conv3x3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels)
        )

        # receptive field 5x5
        self.conv5x5 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels)
        )

        # receptive field 7x7
        self.conv7x7 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
        )

        # receptive field 9*9
        self.conv9x9 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
        )

        self.convs = [self.conv1x1, self.conv3x3, self.conv5x5,
                      self.conv7x7, self.conv9x9]

        self.trans_layers = nn.ModuleList()

        for i in range(len(self.convs)):
            self.trans_layers.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        out_channels, out_channels, kernel_size=1, stride=1, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(
                        out_channels, out_channels, kernel_size=1, stride=1, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels)
                )
            )
        self.trans_layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                out_channels, len(self.convs)*out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(len(self.convs)*out_channels),
        )

        self.eca1x1 = ECALayer(out_channels)
        self.eca3x3 = ECALayer(out_channels)
        self.eca5x5 = ECALayer(out_channels)
        self.eca7x7 = ECALayer(out_channels)
        self.eca9x9 = ECALayer(out_channels)

    def wrap(self, x, F):
        return ME.SparseTensor(
            F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

    def forward(self, x):
        x0 = self.conv1x1(x)
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        x4 = self.conv9x9(x)

        x0 = self.eca1x1(x0)
        x1 = self.eca3x3(x1)
        x2 = self.eca5x5(x2)
        x3 = self.eca7x7(x3)
        x4 = self.eca9x9(x4)

        w0 = self.trans_layers[0](x0)
        w1 = self.trans_layers[1](x1)
        w2 = self.trans_layers[2](x2)
        w3 = self.trans_layers[3](x3)
        w4 = self.trans_layers[4](x4)

        w = w0+w1+w2+w3+w4
        w = self.trans_layer2(w)

        f = w.F.reshape(w.F.shape[0], 5, -1)
        f = F.softmax(f, dim=1).permute(1, 0, 2)
        w0, w1, w2, w3, w4 = self.wrap(w, f[0]), self.wrap(
            w, f[1]), self.wrap(w, f[2]), self.wrap(w, f[3]), self.wrap(w, f[4])
        x = w0*x0+w1*x1+w2*x2+w3*x3+w4*x4

        return x


class TransLoc3DFPN(nn.Module):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, conv0_out_channels, conv0_kernel_size, conv0_stride, conv1_out_channels, conv1_kernel_size, conv1_stride,
                 num_attn_layers, global_channels, local_channels, num_centers, num_heads, out_channels):
        super().__init__()
        self.up_convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.up_blocks = nn.ModuleList()   # Bottom-up blocks
        self.down_convs = nn.ModuleList()   # Top-down tranposed convolutions
        self.skip_conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.up_conv0 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, conv0_out_channels, kernel_size=conv0_kernel_size, stride=conv0_stride, dimension=3),
            ME.MinkowskiBatchNorm(conv0_out_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.up_convs.append(
            nn.Sequential(
                ME.MinkowskiConvolution(conv0_out_channels, conv1_out_channels, kernel_size=conv1_kernel_size, stride=conv1_stride, dimension=3),
                ME.MinkowskiBatchNorm(conv1_out_channels),
                ME.MinkowskiReLU(inplace=True)
            )
        )
        self.up_blocks.append(SelectiveInceptionBlock(conv1_out_channels, global_channels))

        self.transformer = TransformerBlock(num_attn_layers, global_channels, local_channels, num_centers, num_heads)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(global_channels*(num_attn_layers+1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv0(x)

        for idx, (conv, block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x = conv(x)
            x = block(x)

        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        x = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        x = x.permute(0, 2, 1)
        # (batch_size, feature_size, n_points)
        x = self.transformer(x)
        x = self.conv_fuse(x)
        return x, None

