# Code adapted or modified from MinkLoc3DV2 repo: https://github.com/jac99/MinkLoc3Dv2
import numpy as np
from typing import List
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass


class PolarQuantizer(Quantizer):
    def __init__(self, quant_step: List[float]):
        assert len(quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.theta_range = int(360. // self.quant_step[0])
        # self.quant_step = torch.tensor(quant_step, dtype=torch.float)

    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(pc[:, 1], pc[:, 0]) * 180./np.pi  # yaw
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2)  # dist on z=0 planar
        z = pc[:, 2]
        polar_pc = torch.stack([theta, dist, z], dim=1)
        # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
        polar_pc = polar_pc / self.quant_step
        quantized_polar_pc, ndx = ME.utils.sparse_quantize(polar_pc, quantization_size=1., return_index=True)
        # Return quantized coordinates and indices of selected elements
        return quantized_polar_pc, ndx


class CartesianQuantizer(Quantizer):
    def __init__(self, quant_step: float, normalize=False):
        self.quant_step = quant_step
        self.normalize = normalize

    def __call__(self, pc):
        if pc.shape[1] == 5:
            quantized_pc, ndx = ME.utils.sparse_quantize(
                self.normalize_xyz(pc[:, :3]),
                quantization_size=self.quant_step,
                return_index=True,
            )
            quantized_pc = torch.cat((quantized_pc, pc[ndx][:, 3:]), 1)
        else:
            assert pc.shape[1] == 3
            quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coo
        return quantized_pc, ndx

    def normalize_xyz(self, points):
        if not self.normalize:
            return points
        else:
            centroid = torch.mean(points, dim=0)
            points -= centroid
            furthest_distance = torch.max(
                torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=-1))
            )
            points /= furthest_distance

            return points