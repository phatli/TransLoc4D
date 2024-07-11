import torch.nn as nn
import MinkowskiEngine as ME
import torch
from pointops.functions import pointops
from functools import partial

def sparse_to_tensor(tensorIn: ME.SparseTensor, batch_size: int, padding_value: float):

    points, feats = [], []
    for i in range(batch_size):
        points.append(tensorIn.C[tensorIn.C[:, 0] == i, :])
        feats.append(tensorIn.F[tensorIn.C[:, 0] == i, :])

    # padding
    max_len = max([len(i) for i in feats])
    padding_num = [max_len - len(i) for i in feats]
    if padding_value is not None:
        padding_funcs = [
            nn.ConstantPad2d(padding=(0, 0, 0, i), value=padding_value)
            for i in padding_num
        ]

        tensor_feats = torch.stack(
            [pad_fun(e) for e, pad_fun in zip(feats, padding_funcs)], dim=0
        )  # B,N,E
        tensor_coords = torch.stack(
            [pad_fun(e) for e, pad_fun in zip(points, padding_funcs)], dim=0
        )  # B,N,3
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = (
            torch.stack([pad_fun(e) for e, pad_fun in zip(mask, padding_funcs)], dim=0)
            .bool()
            .squeeze(dim=2)
        )  # B,N
    else:  # is None
        tensor_feats = torch.stack(
            [
                torch.cat((feats[i], feats[i][-1].repeat(num, 1)), dim=0)
                for i, num in enumerate(padding_num)
            ],
            dim=0,
        )
        tensor_coords = torch.stack(
            [
                torch.cat((points[i], points[i][-1].repeat(num, 1)), dim=0)
                for i, num in enumerate(padding_num)
            ],
            dim=0,
        )
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = (
            torch.stack(
                [
                    torch.cat((mask[i], torch.zeros(1, 1).repeat(num, 1)), dim=0)
                    for i, num in enumerate(padding_num)
                ],
                dim=0,
            )
            .bool()
            .squeeze(dim=2)
        )  # B,N
    return tensor_feats, tensor_coords, mask


class Interpolate(nn.Module):
    """sparse_tensor  interpolated to xyz_t position, return added f_t"""

    def __init__(
        self,
        inplanes,
        outplanes,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantization_size=1.0,
    ):
        super(Interpolate, self).__init__()
        D = 3
        self.quantization_size = quantization_size
        self.conv1x1 = ME.MinkowskiConvolution(
            inplanes, outplanes, kernel_size=1, stride=1, dimension=D
        )
        # self.conv1x1 = DSConvBlock(inplanes, outplanes, kernel_size=1)

        self.norm = ME.MinkowskiBatchNorm(outplanes)
        self.act = ME.MinkowskiGELU()

        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            # if isinstance(m, ME.MinkowskiBatchNorm):
            #     nn.init.constant_(m.bn.weight, 1)
            #     nn.init.constant_(m.bn.bias, 0)

    def forward(self, sparse, xyz_t):
        # xzy_t is of shape N, 3 and should be transposed to B,3,N
        # sparse is of shape B*N, dim

        # sparse = self.sort_sparse_tensor(sparse)
        # xyz_t = self.sort_sparse_tensor(xyz_t)

        features = sparse.decomposed_features
        coords = sparse.decomposed_coordinates
        xyzs_features = xyz_t.decomposed_features
        xyzs = xyz_t.decomposed_coordinates
        
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        # features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        for i in range(batch_size):
            sparse_feature_single = ME.SparseTensor(
                features=features[i],
                coordinates=coords[i]
            )
            sparse_feature_single = self.sort_sparse_tensor(sparse_feature_single)
            features_single = sparse_feature_single.features.unsqueeze(0)
            coords_single = sparse_feature_single.coordinates.unsqueeze(0)
            
            xyz_single = ME.SparseTensor(
                features=xyzs_features[i],
                coordinates=xyzs[i]
            )
            xyz_single = self.sort_sparse_tensor(xyz_single)
            xyz_t_single = xyz_single.coordinates.unsqueeze(0)
            
            # features_single = features[i].unsqueeze(0)  # 1, num_voxels, 256
            # coords_single = coords[i].unsqueeze(0)  # 1, num_voxels, 3
            # xyz_t_single = xyzs[i].unsqueeze(0)  # 1, num_conv0_voxelx, 3

            features_single = self.interpolate(
                features_single, coords_single, xyz_t_single.float()
            )

            features[i] = features_single[0]
            coords[i] = xyz_t_single[0].float()

        # feat0.features =torch.cat(features,0)
        # feat0.features.data =torch.cat(features,0)

        # indices_sort = np.argsort(self.array2vector(xyz_t.coordinates.cpu(),
        # xyz_t.coordinates.cpu().max()+1))

        # sparse = ME.SparseTensor(
        #     features=torch.cat(features, 0)[indices_sort],
        #     coordinates=xyz_t.coordinates[indices_sort],
        #     coordinate_manager=xyz_t.coordinate_manager,
        # )
        sparse = ME.SparseTensor(
            features=torch.cat(features, 0),
            coordinates=ME.utils.batched_coordinates(coords).to(features_single.device)
        )

        return sparse
    
    def array2vector(self, array, step):
        array, step = array.long(), step.long()
        vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])
        return vector
    
    def sort_sparse_tensor(self, sparse_tensor):
        indices_sort = torch.argsort(self.array2vector(sparse_tensor.C,
        sparse_tensor.C.max()+1))
        sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
        coordinates=sparse_tensor.C[indices_sort],
        tensor_stride=sparse_tensor.tensor_stride[0],
        device=sparse_tensor.device)
        return sparse_tensor_sort

    def interpolate(self, f_tensor, x_tensor, xyz_t):

        # B, N, _ = xyz_t.size()
        B = len(xyz_t)
        # conv1x1
        # sparse = self.act(self.norm(self.conv1x1(sparse)))
        # sparse = self.conv1x1(sparse)
        # -------------------------------------------interpolate------------------------------------#
        # f_tensor, x_tensor, _ = sparse_to_tensor(
        #     sparse, B, 1e3
        # )  # padding B,N,E  B,N,3  B,N
        dist, idx = pointops.nearestneighbor(
            xyz_t, x_tensor[:, :, 0:] * self.quantization_size
        )  # unknown, known B,N,3  x_tensor[:, :, 1:]
        ## dist/idx is of shape 1, num_conv0_voxels, 3
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # (b, c, n)
        interpolated_feats = pointops.interpolation(
            f_tensor.transpose(1, 2).contiguous(), idx, weight
        )  # known_feats
        # f_tensor.transpose(1, 2) is of shape [1, 256, num_voxels]
        # interpolated_feats is of shape [1,256, num_conv0_voxels]
        interpolated_feats = self.act(
            self.ln(interpolated_feats.transpose(1, 2))
        )  # LayerNorm  b,n,c
        # interpolated_feats = interpolated_feats.transpose(1, 2).contiguous()

        return interpolated_feats
