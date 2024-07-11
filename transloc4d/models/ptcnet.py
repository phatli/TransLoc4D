import torch
import torch.nn as nn
from functools import partial

import MinkowskiEngine as ME

from typing import List

from pointops.functions import pointops
from .pt_util import SharedMLP_1d, SharedMLP


def sparse_to_tensor(tensorIn: ME.SparseTensor, batch_size: int, padding_value: float):

    points, feats = [], []
    for i in range(batch_size):
        points.append(tensorIn.C[tensorIn.C[:, 0] == i, :])
        feats.append(tensorIn.F[tensorIn.C[:, 0] == i, :])

    # padding
    max_len = max([len(i) for i in feats])
    padding_num = [max_len - len(i) for i in feats]
    if padding_value is not None:
        padding_funcs = [nn.ConstantPad2d(padding=(0, 0, 0, i), value=padding_value) for i in
                         padding_num]

        tensor_feats = torch.stack([pad_fun(e) for e, pad_fun in zip(feats, padding_funcs)], dim=0)  # B,N,E
        tensor_coords = torch.stack([pad_fun(e) for e, pad_fun in zip(points, padding_funcs)], dim=0)  # B,N,3
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = torch.stack([pad_fun(e) for e, pad_fun in zip(mask, padding_funcs)], dim=0).bool().squeeze(dim=2)  # B,N
    else:  # is None
        tensor_feats = torch.stack(
            [torch.cat((feats[i], feats[i][-1].repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)], dim=0)
        tensor_coords = torch.stack(
            [torch.cat((points[i], points[i][-1].repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)], dim=0)
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = torch.stack(
            [torch.cat((mask[i], torch.zeros(1, 1).repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)],
            dim=0).bool().squeeze(dim=2)  # B,N
    return tensor_feats, tensor_coords, mask


class BasicSpconvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super(BasicSpconvBlock, self).__init__()
        D = 3
        self.conv = ME.MinkowskiConvolution(inplanes, outplanes, kernel_size=kernel_size, stride=stride, bias=False,
                                            dimension=D)
        self.bn = ME.MinkowskiBatchNorm(outplanes, eps=1e-6)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class PTC_Net(nn.Module):

    def __init__(self, params_mink):
        super().__init__()

        self.use_sparse_tensor = False
        self.use_top_down = True

        # --------------------spconv-----------------------#
        self.quantization_size = params_mink.quantization_step
        self.conv0_kernel_size = 5
        base_channel = 32
        stride = 2
        self.kernel_size = 3
        self.SPconv_Fun = BasicSpconvBlock
        # --------------------transformer-----------------------#
        c = 3  # default=3
        self.num_top_down_trans = 4
        self.use_group_trans = True
        groups = [4, 4, 4, 4]
        group_strategy = 'interval'  # interval(default), nearest
        cross_trans = True  # cross transformer
        Group_SA_Fun = Group_SA  # Group_SA: interval+nearest
        FFN = False
        sap = [1024, 256, 64, 16]
        fs = [256, 256, 256, 256]
        gp = 8
        mlp_base_layers = [base_channel, 2 * base_channel]
        mlp_bn = True
        mlp_gp = 1

        print('-------------------',
              '\nSPconv_Fun:', self.SPconv_Fun,
              '\nstride:', stride,
              '\nuse_group_trans:', self.use_group_trans,
              '\ngroup_strategy:', Group_SA_Fun,
              '\ngroups:', groups,
              '\ngp:', gp,
              '\nmlp_gp:', mlp_gp,
              '\ncross_trans:', cross_trans,
              '\nsap:', sap,
              '\nnum_top_down_trans:', self.num_top_down_trans,
              '\n-------------------'
              )
        # ----------------------------------Stem Stage -----------------------------------------------#
        self.spconv0 = BasicSpconvBlock(1, 32, kernel_size=self.conv0_kernel_size, stride=1)  # 1->32

        # ----------------------------------Stage 0 -----------------------------------------------#
        group = groups[0]
        self.spconvs = nn.ModuleList()

        if self.SPconv_Fun is not None:
            self.spconv1 = self.SPconv_Fun(32, 2 * base_channel, kernel_size=self.conv0_kernel_size, stride=1)  # 32->64
        else:
            self.spconv1 = None

        self.spconvs.append(self.spconv1)

        self.sample_inter0 = Sample_interpolated(sap[0], 2 * base_channel, 2 * base_channel,
                                                 group=group, first_sample=True)
        if not self.use_group_trans:
            self.trans_block0 = SA_Layer(2 * base_channel, gp, FFN=FFN)
        else:
            self.trans_block0 = Group_SA_Fun(2 * base_channel, group=group, shared_gp=gp, group_strategy=group_strategy,
                                         cross_trans=cross_trans, FFN=FFN)

        self.mlp0 = SharedMLP_1d(mlp_base_layers, bn=mlp_bn, groups=mlp_gp)  # 32->64

        #----------------------------------Stage 1 --------------------------------------------------#
        group = groups[1]
        stage_1_channel = base_channel * 2

        if self.SPconv_Fun is not None:
            self.spconv1 = self.SPconv_Fun(stage_1_channel, 2 * stage_1_channel, kernel_size=self.kernel_size, stride=stride)  # 64->128
        else:
            self.spconv1 = None

        self.spconvs.append(self.spconv1)

        self.sample_inter1 = Sample_interpolated(sap[1], 2 * stage_1_channel, 2 * stage_1_channel, group=group)
        if not self.use_group_trans:
            self.trans_block1 = SA_Layer(2 * stage_1_channel, gp, FFN=FFN)
        else:
            self.trans_block1 = Group_SA_Fun(2 * stage_1_channel, group=group, shared_gp=gp, group_strategy=group_strategy,
                                         cross_trans=cross_trans, FFN=FFN)

        self.mlp1 = SharedMLP_1d([i * 2 for i in mlp_base_layers], bn=mlp_bn, groups=mlp_gp)  # 64->128
        # ----------------------------------Stage 2 --------------------------------------------------#
        group = groups[2]
        stage_2_channel = base_channel * 4

        if self.SPconv_Fun is not None:
            self.spconv2 = self.SPconv_Fun(stage_2_channel, 2 * stage_2_channel,
                                        kernel_size=self.kernel_size, stride=stride)  # 128->256
        else:
            self.spconv2 = None

        self.spconvs.append(self.spconv2)

        self.sample_inter2 = Sample_interpolated(sap[2], 2 * stage_2_channel, 2 * stage_2_channel, group=group)
        if not self.use_group_trans:
            self.trans_block2 = SA_Layer(2 * stage_2_channel, gp, FFN=FFN)
        else:
            self.trans_block2 = Group_SA_Fun(2 * stage_2_channel, group=group, shared_gp=gp, group_strategy=group_strategy,
                                         cross_trans=cross_trans, FFN=FFN)

        self.mlp2 = SharedMLP_1d([i * 4 for i in mlp_base_layers], bn=mlp_bn, groups=mlp_gp)  # 128->256
        # ----------------------------------Stage 3 --------------------------------------------------#
        group = groups[3]
        stage_3_channel = base_channel * 8

        if self.SPconv_Fun is not None:
            self.spconv3 = self.SPconv_Fun(stage_3_channel, 2 * stage_3_channel,
                                        kernel_size=self.kernel_size, stride=stride)  # 256->512
        else:
            self.spconv3 = None

        self.spconvs.append(self.spconv3)

        self.sample_inter3 = Sample_interpolated(sap[3], 2 * stage_3_channel, 2 * stage_3_channel, group=group)
        if not self.use_group_trans:
            self.trans_block3 = SA_Layer(2 * stage_3_channel, gp, FFN=FFN)
        else:
            self.trans_block3 = Group_SA_Fun(2 * stage_3_channel, group=group, shared_gp=gp, group_strategy=group_strategy,
                                         cross_trans=cross_trans, FFN=FFN)

        self.mlp3 = SharedMLP_1d([i * 8 for i in mlp_base_layers], bn=mlp_bn, groups=mlp_gp)  # 256->512
        # ----------------------------------Top Down ---------------------------------------------------#
        if self.use_top_down:
            self.FP_modules = nn.ModuleList()

            self.FP_modules.append(PointNet2FPModule(mlp=[fs[1] + c, 256, fs[0]]))

            self.FP_modules.append(PointNet2FPModule(mlp=[fs[2] + stage_1_channel, 256, fs[1]], groups=mlp_gp))

            self.FP_modules.append(PointNet2FPModule(mlp=[fs[3] + stage_2_channel, 256, fs[2]], groups=mlp_gp))

            self.FP_modules.append(PointNet2FPModule(mlp=[2 * stage_3_channel + stage_3_channel, 256, fs[3]], groups=mlp_gp))

        self.Gem = GeM()

    def forward(self, batch):
        feature_maps_sparse = []
        feature_maps_t = []
        xyz_ts = []
        # -------------------------------Stem stage---------------------------------#
        sparse_tensor = ME.SparseTensor(batch['features'], coordinates=batch['coords'])

        if self.use_sparse_tensor:
            B = sparse_tensor.C[-1, 0].item() + 1
            _, bxyz_tensor, _ = sparse_to_tensor(sparse_tensor, B, padding_value=None)
            xyz_tensor = bxyz_tensor[:, :, 1:] * self.quantization_size
        else:
            xyz_tensor = batch['batch']

        feats_t = xyz_tensor.transpose(1, 2).contiguous()  # 4096,3
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_tensor)

        # spconv-------------------------------------------------------------------#
        sparse = self.spconv0(sparse_tensor)  # 32
        feature_maps_sparse.append(sparse)
        # bottom up
        if self.SPconv_Fun is not None:
            for spconv in self.spconvs:
                sparse = spconv(sparse)
                feature_maps_sparse.append(sparse)
        else:
            for spconv in self.spconvs:
                feature_maps_sparse.append(None)

        if self.SPconv_Fun is not None:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, feature_maps_sparse[1], None)  # 1024,64
        else:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, sparse, None)  # 1024,64

        feats_t = self.trans_block0(inter_feats_t)  # 1024,64

        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)
        # ---------------------------------stage 1 --------------------------------- #
        xyz_t, inter_feats_t, feats_t = self.sample_inter1(xyz_t, feature_maps_sparse[2], feats_t)  # 256,128/256,64

        feats_t = self.mlp1(feats_t)  # 256,128

        # inter_feats_t = None
        feats_t = self.trans_block1(feats_t, inter_feats_t)  # 256,128
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)
        # ---------------------------------stage 2 --------------------------------- #
        xyz_t, inter_feats_t, feats_t = self.sample_inter2(xyz_t, feature_maps_sparse[3], feats_t)  # 64,256/64,128

        feats_t = self.mlp2(feats_t)  # 64,256

        # inter_feats_t = None
        feats_t = self.trans_block2(feats_t, inter_feats_t)  # 64,256
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)
        # ---------------------------------stage 3 --------------------------------- #
        xyz_t, inter_feats_t, feats_t = self.sample_inter3(xyz_t, feature_maps_sparse[4], feats_t)  # 16,512/16,256

        feats_t = self.mlp3(feats_t)  # 16,512

        feats_t = self.trans_block3(feats_t, inter_feats_t)  # 16,512
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)

        if self.use_top_down:
            # transformer
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                feature_maps_t[i - 1] = self.FP_modules[i](xyz_ts[i - 1], xyz_ts[i], feature_maps_t[i - 1],
                                                           feature_maps_t[i])  # up-down
            feats_t = feature_maps_t[len(self.FP_modules) - self.num_top_down_trans]

        out = self.Gem(feats_t).squeeze(-1)

        return {'global': out}

class PTC_Net_L(nn.Module):

    def __init__(self, params_mink):
        super().__init__()

        self.use_sparse_tensor = params_mink.use_sparse_tensor
        self.use_top_down = True
        # --------------------spconv-----------------------#
        self.quantization_size = params_mink.quantization_step
        self.conv0_kernel_size = 5
        base_channel = 16
        stride = 2
        self.kernel_size = 3
        self.SPconv_Fun = None  # BasicSpconvBlock, None
        # --------------------transformer-----------------------#
        c = 3  # default=3
        self.num_top_down_trans = 1
        self.use_group_trans = True
        groups = [4, 4, 4, 4]
        group_strategy = 'interval'  # interval(default), nearest
        cross_trans = True  # cross transformer
        Group_SA_Fun = Group_SA  # Group_SA: interval+nearest
        FFN = False
        sap = [1024, 256, 64, 16]
        fs = [256, 256, 256, 256]
        gp = 8
        mlp_base_layers = [base_channel, 2 * base_channel]
        mlp_bn = True
        mlp_gp = 1

        # print('-------------------',
        #       '\nSPconv_Fun:', self.SPconv_Fun,
        #       '\nstride:', stride,
        #       '\nuse_group_trans:', self.use_group_trans,
        #       '\ngroup_strategy:', Group_SA_Fun,
        #       '\ngroups:', groups,
        #       '\ngp:', gp,
        #       '\nmlp_gp:', mlp_gp,
        #       '\ncross_trans:', cross_trans,
        #       '\nsap:', sap,
        #       '\nnum_top_down_trans:', self.num_top_down_trans,
        #       '\n-------------------'
        #       )
        # ----------------------------------Stem Stage -----------------------------------------------#
        self.spconv0 = BasicSpconvBlock(1, 32, kernel_size=self.conv0_kernel_size, stride=1)  # 1->32

        # ----------------------------------Stage 0 -----------------------------------------------#
        group = groups[0]
        self.spconvs = nn.ModuleList()

        if self.SPconv_Fun is not None:
            self.spconv1 = self.SPconv_Fun(32, 2 * base_channel, kernel_size=self.conv0_kernel_size, stride=1)  # 32->64
        else:
            self.spconv1 = None

        self.spconvs.append(self.spconv1)

        self.sample_inter0 = Sample_interpolated(sap[0], 2 * base_channel, 2 * base_channel,
                                                 group=group, first_sample=True)
        if not self.use_group_trans:
            self.trans_block0 = SA_Layer(2 * base_channel, gp, FFN=FFN)
        else:
            self.trans_block0 = Group_SA_Fun(2 * base_channel, group=group, shared_gp=gp, group_strategy=group_strategy,
                                             cross_trans=cross_trans, FFN=FFN)

        self.mlp0 = SharedMLP_1d(mlp_base_layers, bn=mlp_bn, groups=mlp_gp)  # 32->64

        # ----------------------------------Top Down ---------------------------------------------------#
        if self.use_top_down:
            self.FP_modules = nn.ModuleList()
            self.FP_modules.append(PointNet2FPModule(mlp=[32 + c, 256, fs[0]]))  # PTC-Net-L

        self.Gem = GeM()

    def forward(self, batch):
        # feature maps
        feature_maps_sparse = []
        feature_maps_t = []
        xyz_ts = []
        # -------------------------------Stem stage---------------------------------#
        sparse_tensor = ME.SparseTensor(batch['features'], coordinates=batch['coords'])

        if self.use_sparse_tensor:
            B = sparse_tensor.C[-1, 0].item() + 1
            _, bxyz_tensor, _ = sparse_to_tensor(sparse_tensor, B, padding_value=None)
            xyz_tensor = bxyz_tensor[:, :, 1:] * self.quantization_size
        else:
            xyz_tensor = batch['batch'] #B,4096,3

        feats_t = xyz_tensor.transpose(1, 2).contiguous()  # B,4096,3  -->  B,3,4096
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_tensor)

        # spconv-------------------------------------------------------------------#
        sparse = self.spconv0(sparse_tensor)  # sparse_tensor[~B*N,1]  -->  sparse[~B*N,32]
        feature_maps_sparse.append(sparse)
        # bottom up
        if self.SPconv_Fun is not None:
            for spconv in self.spconvs:
                sparse = spconv(sparse)
                feature_maps_sparse.append(sparse)
        else:
            for spconv in self.spconvs:
                feature_maps_sparse.append(None)

        if self.SPconv_Fun is not None:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, feature_maps_sparse[1], None)  # 1024,64
        else:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, sparse, None)  
            # (xyz_tensor[B,4096,3], sparse[~B*N,32]) --> (xyz_t[B,1024,3], inter_feats_t[B,32,1024]) 

        feats_t = self.trans_block0(inter_feats_t)  # 1024,64 #inter_feats_t [B,32,1024] --> feats_t [B,32,1024]

        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)

        if self.use_top_down:
            # transformer
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                feature_maps_t[i - 1] = self.FP_modules[i](xyz_ts[i - 1], xyz_ts[i], feature_maps_t[i - 1],
                                                           feature_maps_t[i])  # up-down
            feats_t = feature_maps_t[len(self.FP_modules) - self.num_top_down_trans] # feat_t:torch.Size([96, 256, 4096])

        out = self.Gem(feats_t).squeeze(-1) # out:torch.Size([96, 256])

        return {'global': out}
    
class FTU(nn.Module):
    """ sparse_tensor  interpolated to xyz_t position, return added f_t
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), quantization_size=0.01):
        super(FTU, self).__init__()
        D = 3
        self.quantization_size = quantization_size
        self.conv1x1 = ME.MinkowskiConvolution(inplanes, outplanes, kernel_size=1, stride=1, dimension=D)
        # self.conv1x1 = DSConvBlock(inplanes, outplanes, kernel_size=1)

        self.norm = ME.MinkowskiBatchNorm(outplanes)
        self.act = ME.MinkowskiGELU()

        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            # if isinstance(m, ME.MinkowskiBatchNorm):
            #     nn.init.constant_(m.bn.weight, 1)
            #     nn.init.constant_(m.bn.bias, 0)

    def forward(self, sparse, xyz_t):
        B, N, _ = xyz_t.size()
        # conv1x1
        # sparse = self.act(self.norm(self.conv1x1(sparse)))
        sparse = self.conv1x1(sparse)
        # -------------------------------------------interpolate------------------------------------#
        f_tensor, x_tensor, _ = sparse_to_tensor(sparse, B, 1e3)  # padding B,N,E  B,N,3  B,N
        dist, idx = pointops.nearestneighbor(xyz_t, x_tensor[:, :,
                                                    1:] * self.quantization_size)  # unknown, known B,N,3  x_tensor[:, :, 1:]
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # (b, c, n)
        interpolated_feats = pointops.interpolation(f_tensor.transpose(1, 2).contiguous(), idx,
                                                    weight)  # known_feats
        interpolated_feats = self.act(self.ln(interpolated_feats.transpose(1, 2)))  # LayerNorm  b,n,c
        interpolated_feats = interpolated_feats.transpose(1, 2).contiguous()

        return interpolated_feats

class Sample_interpolated(nn.Module):
    def __init__(self, npoint, inplanes, outplanes, group=4, first_sample=False):  # fps, nearest
        super(Sample_interpolated, self).__init__()
        self.npoint = npoint
        self.get_interpolate = FTU(inplanes, outplanes)
        self.first_sample = first_sample

    def forward(self, xyz, sparse, feat_t=None):
        r"""
        Parameters
        ----------
        xyz :
            (B, N, 3) tensor of the xyz coordinates of the features
        sparse :
            (~B*N, C) tensor of the descriptors of the the sparse features
        Returns
        -------
        new_xyz :
            (B, npoint, 3) tensor of the new features' xyz
        new_feat :
            (B, npoint, C) tensor of the new_features descriptors
        """

        # ---step.1 get sample xyz---------------------------------- #
        B, _, N = xyz.shape
        if self.first_sample:
            xyz_trans = xyz.transpose(1, 2).contiguous()  # B x 3 x N
            center_idx = pointops.furthestsampling(xyz, self.npoint)  # B,npoint
            new_xyz = pointops.gathering(
                xyz_trans,
                center_idx
            ).transpose(1, 2).contiguous() if self.npoint is not None else None  # b,n,3
            if feat_t is not None:
                new_features = pointops.gathering(feat_t, center_idx)
            else:
                new_features = None
        else:
            new_xyz = xyz[:, :self.npoint, :].contiguous()
            if feat_t is not None:
                new_features = feat_t[:, :, :self.npoint]
            else:
                new_features = None

        # ---step.2 get interpolated sparse features---------------- #
        if sparse is not None:
            interpolated_feat = self.get_interpolate(sparse, new_xyz)
        else:
            interpolated_feat = None
        return new_xyz, interpolated_feat, new_features

class SA_Layer(nn.Module):
    def __init__(self, inchannels, gp=1, FFN=True):  # feature dim
        super().__init__()
        mid_channels = inchannels
        outchannels = inchannels
        self.gp = gp
        self.FFN = FFN
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(inchannels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(inchannels, mid_channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(inchannels, inchannels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.layerNorm1 = nn.LayerNorm(inchannels, eps=1e-6)
        self.act = nn.ReLU(inplace=True)
        if self.FFN:
            self.layerNorm2 = nn.LayerNorm(inchannels, eps=1e-6)

            self.fc1 = nn.Linear(inchannels, inchannels * 4)
            self.fc2 = nn.Linear(inchannels * 4, outchannels)

    def forward(self, x, x2=None):
        r"""
        x: B x C x N
        x2:  B x C x N
        """
        bs, ch, nums = x.size()
        residual = x
        if x2 is not None:
            x = x + x2

        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch // self.gp, nums)
        x_q = x_q.permute(0, 1, 3, 2)  # B x gp x num x C'

        x_k = self.k_conv(x)  # B x C x N
        x_k = x_k.reshape(bs, self.gp, ch // self.gp, nums)  # B x gp x C' x nums

        x_v = self.v_conv(x)  # B x C x N
        energy = torch.matmul(x_q, x_k)  # B x gp x N x N

        # energy = torch.sum(energy, dim=1, keepdims=False) / (ch ** 0.5)  # B,N,N
        energy = torch.sum(energy, dim=1, keepdims=False)  # B,N,N

        attn = self.softmax(energy)

        x_r = torch.matmul(x_v, attn) / (ch ** 0.5)  # B,N,C

        x = self.layerNorm1((residual + x_r).transpose(1, 2))  # B,N,C

        if self.FFN:
            x = x + self.fc2(self.act(self.fc1(x)))
            x = self.act(self.layerNorm2(x)).transpose(1, 2).contiguous()  # B,C,N
        else:
            x = x.transpose(1, 2).contiguous()  # B,C,N
        return x

class Group_SA(nn.Module):

    def __init__(self, inplanes, group=4, shared_gp=1, group_strategy='interval',  # interval, nearest
                 cross_trans=False, FFN=True):
        super(Group_SA, self).__init__()
        self.trans_block = SA_Layer(inplanes, gp=shared_gp, FFN=FFN)
        self.gp = group
        self.group_strategy = group_strategy
        self.cross_trans = cross_trans
        if cross_trans:
            if self.gp != 1:
                self.cross_trans_block2 = Cross_SA_Layer(inplanes, shared_gp=shared_gp, group=group, FFN=FFN)
            else:
                self.cross_trans_block2 = SA_Layer(inplanes, gp=shared_gp, FFN=FFN)

    def forward(self, feat, feat2=None, interval=True):
        # ---step.1 interval group feat---------------------------------- #
        B, C, N = feat.shape  # B x C x N
        assert N % self.gp == 0
        group_feat = []
        if self.gp == 1:
            group_feat = feat  # B x C x N
            group_feat2 = feat2
        else:
            ind = torch.arange(0, N, self.gp, dtype=torch.int64, device="cuda")
            feat = feat.transpose(1, 2)  # B x N x C
            for i in range(self.gp):
                group_feat.append(torch.index_select(feat, 1, ind + i))
            group_feat = torch.vstack(group_feat)  # gp*B,N/gp,C
            group_feat = group_feat.transpose(1, 2)  # gp*B,C,N/gp
            if feat2 is not None:
                group_feat2 = []
                feat2 = feat2.transpose(1, 2)
                for i in range(self.gp):
                    group_feat2.append(torch.index_select(feat2, 1, ind + i))
                group_feat2 = torch.vstack(group_feat2)
                group_feat2 = group_feat2.transpose(1, 2)
            else:
                group_feat2 = None
        # ---step.2 do SA-------------------------------------- #
        feat = self.trans_block(group_feat, group_feat2)  # gp*B,C,N/gp
        # ---step.3 ungroup feat-------------------------------------- #
        group_feat_new = []
        if self.gp == 1:
            group_feat_new = feat  # B x C x N
        else:
            group_feat = feat.transpose(1, 2)  # gp*B,N/gp,C
            for i in range(self.gp):
                group_feat_new.append(group_feat[B * i:B + B * i, :])
            group_feat_new = torch.cat(group_feat_new, dim=2).view(B, N, C)  # B,N,C
            # group_feat_new = group_feat_new.transpose(1, 2).contiguous()  # B,C,N

        # ---step.4 nearest group feat---------------------------------- #
        if self.cross_trans and self.gp != 1:
            group_point = N // self.gp
            ind = torch.arange(0, group_point, 1, dtype=torch.int64, device="cuda")
            group_feat = []
            for i in range(self.gp):
                group_feat.append(torch.index_select(group_feat_new, 1, ind + i * group_point))  # gp*B,N/gp,C
            group_feat = torch.vstack(group_feat).transpose(1, 2)  # gp*B,C,N/gp,

            # ---step.5 do cross SA---------------------------------#
            feat = self.cross_trans_block2(group_feat)  # gp*B,C,N/gp
            # feat = self.cross_trans_block2(feat)  # gp*B,C,N/gp
            # feat = group_feat  # gp*B,C,N/gp
            # ---step.6 ungroup feat------------------------------- #
            feat = feat.transpose(1, 2)  # gp*B,N/gp, C
            group_feat_new = []
            for i in range(B):
                for j in range(self.gp):
                    group_feat_new.append(feat[B * j + i])
            group_feat_new = torch.vstack(group_feat_new).reshape(B, N, C).transpose(1, 2).contiguous()  # B,C,N
        elif self.gp == 1:
            group_feat_new = self.cross_trans_block2(group_feat_new)  # B x C x N

        return group_feat_new

class Cross_SA_Layer(nn.Module):  # first KV, last Q

    def __init__(self, inchannels, shared_gp, group, FFN=True):  # feature dim
        super(Cross_SA_Layer, self).__init__()
        mid_channels = inchannels
        outchannels = inchannels
        self.shared_gp = shared_gp
        self.group = group
        assert mid_channels % 4 == 0
        # last group
        self.q_conv = nn.Conv1d(inchannels, mid_channels, 1, bias=False, groups=shared_gp)
        # first group
        self.k_convs = nn.Conv1d(inchannels, mid_channels, 1, bias=False, groups=shared_gp)
        self.v_convs = nn.Conv1d(inchannels, inchannels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.mlp = pt_util.SharedMLP_1d([(group - 1) * inchannels, inchannels, inchannels], bn=True)
        # self.after_norm = nn.BatchNorm1d(inchannels)

        self.layerNorm1 = nn.LayerNorm(inchannels, eps=1e-6)
        self.act = nn.ReLU(inplace=True)

        self.FFN = FFN
        if self.FFN:
            self.layerNorm2 = nn.LayerNorm(inchannels, eps=1e-6)
            self.fc1 = nn.Linear(inchannels, inchannels * 4)
            self.fc2 = nn.Linear(inchannels * 4, outchannels)

    def forward(self, x):
        r"""
        x: gp*B x C x N(N/group)
        """
        bs, ch, nums = x.size()  # gp*B x C x N
        assert bs % self.group == 0
        batch = bs // self.group  # B

        # step.1 get  groups----------------------------------------------#
        first_group = x[0:batch, :]  # 1 * BxCxN
        last_group = x[batch:, :]  # (group-1)*BxCxN
        residual = first_group  # 1*BxCxN

        # step.2 calcu QKV----------------------------------------------#
        x_qs = self.q_conv(last_group)  # (group-1)*BxCxN
        x_qs = x_qs.reshape((self.group - 1) * batch, self.shared_gp, ch // self.shared_gp, nums)
        x_qs = x_qs.permute(0, 1, 3, 2)  # (group-1)*B x gp x N x C'

        x_k = self.k_convs(first_group)  # 1*BxCxN
        x_k = (x_k.reshape(batch, self.shared_gp, ch // self.shared_gp, nums))  # 1* B x gp x C' x N

        x_v = self.v_convs(first_group)  # 1*BxCxN

        # step.3 calcu attns-------------------------------------------#
        x_rs = []
        for i in range(self.group - 1):
            energy = torch.matmul(x_qs[batch * i:batch * i + batch], x_k)  # B x gp x N x N
            energy = torch.sum(energy, dim=1, keepdims=False)  # Bx N x N

            attn = self.softmax(energy)  # 考虑在这加padding, BxNxN
            x_r = torch.matmul(x_v, attn) / (ch ** 0.5)  # BxCxN
            x_rs.append(x_r)  # (group-1)*BxCxN
        x_rs = torch.vstack(x_rs)  # (group-1)*BxCxN
        x_r = torch.mean(x_rs.view(batch, -1, ch, nums), dim=1, keepdim=False)  # BxCxN
        x = self.layerNorm1((residual + x_r).transpose(1, 2))  # 1*BxNxC

        if self.FFN:
            x = x + self.fc2(self.act(self.fc1(x)))  # BxCxN
            x = self.act(self.layerNorm2(x)).transpose(1, 2).contiguous()  # BxCxN
        else:
            x = x.transpose(1, 2).contiguous()  # BxCxN
        x = torch.cat([x, last_group], dim=0)  # gp*BxCxN
        return x


class PointNet2FPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True, groups: int = 1):
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn, groups=groups)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor,
                known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features, next layer
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features, current layer
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to, next layer
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated, current layer
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx,
                                                        weight)  # upper features interpolate to current layer
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        # This implicitly applies ReLU on x (clamps negative values)
        x = x.clamp(min=self.eps).pow(self.p)

        x = self.f(x).pow(1. / self.p)
        return x  # Return (batch_size, n_features) tensor
