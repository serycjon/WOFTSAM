# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
import einops
import cv2
import os
import torch
import torch.nn.functional as F
import kornia
import kornia.geometry.conversions as kgc
import kornia.geometry.homography as kgh
# from kornia.utils import _extract_device_dtype
import tqdm
from pathlib import Path
import json
import datetime
import logging
logger = logging.getLogger(__name__)

import flatsam.utils.geom as gu


def find_homography_nonhomogeneous_QR(points1, points2, weights=None):
    r"""Compute the homography matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.
    Using a nonhomogenous system of equations (fixing the bottom-right element of H - H_{3,3} = 1.
    Limitation: unable to estimate H with H_{3,3} = 0, which can happen.
    see Hartley, Zisserman; Multiple View Geometry, 2nd edition Sec. 4.1.2, Example 4.1 on page 90.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)

    # device, dtype = _extract_device_dtype([points1, points2])

    eps = 1e-8
    points1_norm, transform1 = kornia.geometry.epipolar.normalize_points(points1)
    points2_norm, transform2 = kornia.geometry.epipolar.normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1], dim=-1)  # batch, N, 8
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1], dim=-1)   # batch, N, 8
    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(torch.cat((ax, ay), dim=-1), 'B N (two eight) -> B (N two) eight', two=2, eight=8)
    # A_orig = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])   # batch, 2xN, 8
    # assert torch.all(A == A_orig)

    bx = -y2  # batch, N, 1
    by = x2   # batch, N, 1
    b = einops.rearrange(torch.cat((bx, by), dim=-1), 'B N (two one) -> B (N two) one', two=2, one=1)

    if weights is not None:
        # to really be the least squares weights, we would have to use square root of w
        # (to optimize \sum_i w_i (A x_i - b_i)^2, we should multiply both A and b by a diagonal matrix containing sqrt(w_i))

        # splice the weights to each of the two equations (each weight repeated 2 times)
        w = einops.repeat(weights, 'B N -> B (N repeat) 1', repeat=2)
        A = w * A
        b = w * b

    res = torch.linalg.qr(A)
    # now we need to solve Rx = Q^T b
    # res.Q has shape (batch, 2xN, 8)
    lhs = res.R
    rhs = einops.rearrange(res.Q, 'B twoN eight -> B eight twoN', eight=8) @ b

    # res = torch.triangular_solve(rhs, lhs)
    # solution = res.solution  # batch, 8, 1
    solution = torch.linalg.solve_triangular(lhs, rhs, upper=True)
    # add the H_{3,3} = 1 element
    solution = torch.cat([solution, torch.ones((solution.shape[0], 1, 1), dtype=solution.dtype, device=solution.device)],
                         dim=1)
    H = einops.rearrange(solution, 'B (rows cols) 1 -> B rows cols',
                         rows=3, cols=3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm

def torch_proj_errors(GT_H, pts_A, pts_B):
    """Compute L2 distance between given correspondences and correspondences created by H-warp

    args:
        GT_H: (B, 3, 3) batch of GT homographies mapping pts_A somewhere
        pts_A: (B, 3, N) batch of N source points
        pts_A: (B, 3, N) batch of N destination points
    returns:
        L2_err: (B, N) L2 distances
    """
    # proj forward pts_A by GT_H, measure L2 errors
    proj_pts = torch.matmul(GT_H, kgc.convert_points_to_homogeneous(pts_A.permute(0, 2, 1)).permute(0, 2, 1))
    proj_pts = kgc.convert_points_from_homogeneous(proj_pts.permute(0, 2, 1)).permute(0, 2, 1)
    L2_err = torch.sqrt(einops.reduce(torch.square(proj_pts - pts_B),
                                      'B xy N -> B N', xy=2, reduction='sum'))
    return L2_err
