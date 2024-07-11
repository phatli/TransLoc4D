'''
TransVLAD3d
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.neighbors import NearestNeighbors

import h5py

from os.path import join, exists


import MinkowskiEngine as ME


class TransVLAD3d(nn.Module):
    """TransVLAD3d"""

    def __init__(self, dim=128, num_clusters=64, ghost_weighting=True, semantic_init=False, num_shadows=4, num_sc=3, param_norm=True,
                 normalize_input=True, vladv2=False, core_loop=True, num_ghost=9):
        """
        Args:1
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
            pyramid_level: int
                Num of pyramid layers, the k^th layer consists of 4**(k-1) if overlap == False
            Overlap: bool
                If true, using overlapping pyramid vlad pooling on the input feature maps
            Semantic_init: bool
                If true, applying semantic constrained initialization
            Param_norm: bool
                If true, applying parametric normalization
        """
        super(TransVLAD3d, self).__init__()

        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input

        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.core_loop = core_loop  # slower than non-looped, but lower memory usage

        # parametric normalization
        self.param_norm = param_norm
        if self.param_norm == True:
            self.cluster_weights = nn.Parameter(torch.ones(num_clusters))

        # ghost weighting
        self.ghost_weighting = ghost_weighting
        self.semantic_init = semantic_init
        if self.ghost_weighting == False:
            num_ghost = 0
        self.num_ghost = num_ghost
        self.conv = nn.Conv2d(dim, num_clusters + num_ghost,
                              kernel_size=(1, 1), bias=vladv2)
    
    def init_params(self, clsts, traindescs, shadowclsts=None):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight.data[:-self.num_ghost] = nn.Parameter(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))  # (64, 512, 1, 1)
            self.conv.bias = None
            print('init_params done')
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight.data[:-self.num_ghost] = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias.data[:-self.num_ghost] = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

        # if (self.ghost_weighting == True) and (self.semantic_init == True):
        #     assert shadowclsts is not None, "try to implement semantic constrained initialization but cannot find the shadowclsts"

        #     shadowclstsAssign = shadowclsts / \
        #         np.linalg.norm(shadowclsts, axis=1, keepdims=True)
        #     shadowclsts = torch.from_numpy(shadowclstsAssign)

        #     shadowconv_weight_sc = nn.Parameter(torch.from_numpy(
        #         self.alpha*shadowclstsAssign).unsqueeze(2).unsqueeze(3))
        #     if self.conv.bias is not None:
        #         shadowconv_bias_sc = nn.Parameter(
        #             - self.alpha * torch.from_numpy(shadowclsts).norm(dim=1))
        # else:
        # initialize the conv_shadow with present conv to minic vlad
        for i in range(self.num_ghost):
            # initialize the shadows of ith cluster
            idx_list = range(self.num_clusters)
            idx = np.random.choice(idx_list, self.num_ghost, False)
            shadowconv_weight_sc = self.conv.weight.data[idx]
            if self.conv.bias is not None:
                shadowconv_bias_sc = self.conv.bias.data[idx]

        self.conv.weight.data[self.num_clusters:] = shadowconv_weight_sc
        if self.conv.bias is not None:
            self.conv.bias.data[self.num_clusters:] = shadowconv_bias_sc

    def _init_params(self):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, self.traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(self.clsts))
            self.conv.weight.data[:-self.num_ghost] = nn.Parameter(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))  # (64, 512, 1, 1)
            self.conv.bias = None
            print('init_params done')
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(self.traindescs)
            dsSq = np.square(knn.kneighbors(self.clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(self.clsts))

            self.conv.weight.data[:-self.num_ghost] = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias.data[:-self.num_ghost] = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

        if (self.ghost_weighting == True) and (self.semantic_init == True):
            assert self.shadowclsts is not None, "try to implement semantic constrained initialization but cannot find the shadowclsts"

            shadowclstsAssign = self.shadowclsts / \
                np.linalg.norm(self.shadowclsts, axis=1, keepdims=True)
            shadowclsts = torch.from_numpy(shadowclstsAssign)

            shadowconv_weight_sc = nn.Parameter(torch.from_numpy(
                self.alpha*shadowclstsAssign).unsqueeze(2).unsqueeze(3))
            if self.conv.bias is not None:
                shadowconv_bias_sc = nn.Parameter(
                    - self.alpha * torch.from_numpy(shadowclsts).norm(dim=1))
        else:
            # initialize the conv_shadow with present conv to minic vlad
            for i in range(self.num_ghost):
                # initialize the shadows of ith cluster
                idx_list = range(self.num_clusters)
                idx = np.random.choice(idx_list, self.num_ghost, False)
                shadowconv_weight_sc = self.conv.weight.data[idx]
                if self.conv.bias is not None:
                    shadowconv_bias_sc = self.conv.bias.data[idx]

        self.conv.weight.data[self.num_clusters:] = shadowconv_weight_sc
        if self.conv.bias is not None:
            self.conv.bias.data[self.num_clusters:] = shadowconv_bias_sc

    def wrapper(self, x: ME.SparseTensor):
        # x is (batch_size, C, H, W)
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        # batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros
        return features
        # x = self.net_vlad(features)
        # assert x.shape[0] == batch_size
        # return x    # Return (batch_size, output_dim) tensor

    def forward(self, x, vis_kmaps=None, forTNSE=False):

        x = self.wrapper(x).transpose(1,2).unsqueeze(-1) # 110409,256 ==> batch_size, channels=256,num_points(max in minibatch),  1

        N, C, H, W = x.shape 

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x)  # N, K+n_g, H, W
        soft_assign = F.softmax(soft_assign, dim=1)[
            :, :self.num_clusters, :, :]  # N, K, H, W
        # print('soft_assign',soft_assign.shape)('soft_assign', (1, 64, 30, 40))

        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        # calculate residuals to each clusters
        if self.core_loop == True:  # slower than non-looped, but lower memory usage
            for C_idx in range(self.num_clusters):
                residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
                    self.centroids[C_idx:C_idx+1, :].expand(W, -1, -1).expand(H, -1, -1, -1).permute(
                        2, 3, 0, 1).unsqueeze(0)  # N,1,C,H,W - 1,1,C,H,W = N,1,C,H,W
                # print("residual",residual.shape)('residual', (1, 1, 512, 30, 40))
                # print("centroids", self.centroids.shape)('centroids', (64, 512))
                # N,1,C,H,W * N,1,1,H,W = N,1,C,H,W
                residual *= soft_assign[:, C_idx:C_idx+1, :].unsqueeze(2)
                vlad[:, C_idx:C_idx+1,
                        :] = F.normalize(residual.view(N, 1, C, -1).sum(dim=-1), p=2, dim=2)
        else:
            print('to be completed')

        # parametric normalization
        vlad = self.normalization(vlad)
        return vlad

    def normalization(self, vlad):
        ''' vlad: N, C*K'''
        N = vlad.shape[0]
        # parametric normalization
        if self.param_norm == True:
            cluster_weights = self.cluster_weights.expand(N, -1)
            cluster_weights = F.normalize(cluster_weights, p=2, dim=1)  # N,K
            vlad = vlad*cluster_weights.unsqueeze(2)
        vlad = vlad.view(N, -1)  # flatten  of shape N*(K*C)
        # L2 normalize       #output N*(K*C) tensor
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

def find_initcache(pool_layer, dataPath='../data', pooling='transvlad3d', num_clusters=64):
    suffix = '.hdf5'
    backbone = 'MinklocFPN'
    if pooling.lower() in ['appsvr', 'appsvrv2', 'isapvlad', 'isapvladv2', 'cahir', 'patchcahir']:
        initcache = join(dataPath, 'centroids',
                            backbone + '_' + str(num_clusters) + '_desc_cen_' + pooling + suffix)
    else:
        initcache = join(dataPath, 'centroids',
                            backbone + '_' + str(num_clusters) + '_desc_cen'+suffix)

    if not exists(initcache):
        print(
            f"Could not find initialized cluster {initcache}. Won\'t load centroids.")
        # get_clusters()

    assert exists(initcache), 'still could not find initialized cluster'

    print(
        f"Found initialized cluster {initcache}. Loading centroids.")
    with h5py.File(initcache, mode='r') as h5:
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        pool_layer.init_params(clsts, traindescs)
        # pooling.clsts = clsts
        # pooling.traindescs = traindescs
        # pooling._init_params()
        del clsts, traindescs
        print(
            'The pooling layer is first to be intialized from clusters and sampled features.')
