import copy
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .linearselfatt import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear", dropout_p=0.05):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.act = nn.ReLU()

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(
            message.view(bs, -1, self.nhead * self.dim) - x
        )  # [N, L, C]
        message = self.norm1(message)
        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = 256
        self.nhead = 2
        self.layer_names = ["self"] * 1
        encoder_layer = LoFTREncoderLayer(256, self.nhead, "linear")
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        features = feat0.decomposed_features
        batch_size = len(features)

        for i in range(batch_size):
            features_single = features[i].unsqueeze(0)
            assert self.d_model == features_single.size(
                2
            ), "the feature number of src and transformer must be equal"

            for layer, name in zip(self.layers, self.layer_names):
                features_single = layer(features_single, features_single, mask0, mask0)

            features[i] = features_single[0]


        feat0 = ME.SparseTensor(
            features=torch.cat(features, 0),
            coordinates=feat0.coordinates,
            coordinate_manager=feat0.coordinate_manager,
        )

        return feat0
