import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra.utils as hydra_utils
import numpy as np
import sklearn
import sklearn.metrics
import math

from pathlib import Path
import os


def rotate_tensor(im, angles):
    x_mid = (im.size(2) + 1) / 2.
    y_mid = (im.size(3) + 1) / 2.
    # Calculate rotation with inverse rotation matrix
    #angle_ind = angles.long() // 5 + 72
    #rot_matrix = rotmats[angle_ind]
    #rot_matrix = torch.cat([rotmats[angle.long().item()] for angle in angles],0)
    rot_matrix = torch.cat([
        torch.tensor([[
            torch.cos(angle * np.pi / 180.0),
            torch.sin(angle * np.pi / 180.0)
        ],
                      [
                          -1.0 * torch.sin(angle * np.pi / 180.0),
                          torch.cos(angle * np.pi / 180.0)
                      ]]).unsqueeze(0) for angle in angles
    ], 0)

    # Use meshgrid for pixel coords
    xv, yv = torch.meshgrid(torch.arange(im.size(2)), torch.arange(im.size(3)))
    #ind = ind.contiguous().view(-1, 1)
    xv = xv.contiguous().view(-1, 1)
    yv = yv.contiguous().view(-1, 1)
    src_ind = torch.cat(((xv.float() - x_mid), (yv.float() - y_mid)), dim=1)
    src_ind = src_ind.unsqueeze(0).repeat(im.size(0), 1, 1)
    orig_src_ind = torch.cat((xv.float(), yv.float()), dim=1)
    orig_src_ind = orig_src_ind.unsqueeze(0).repeat(im.size(0), 1, 1)

    # Calculate indices using rotation matrix
    #src_ind = torch.matmul(src_ind, rot_matrix.t())
    src_ind = torch.bmm(src_ind, rot_matrix.permute(0, 2, 1))
    src_ind = torch.round(src_ind)
    src_ind += torch.tensor([[[x_mid, y_mid]]])

    batch_i = torch.arange(im.size(0)).unsqueeze(1).unsqueeze(2).repeat(
        1, src_ind.shape[1], 1).view(-1)
    xv = orig_src_ind[:, :, 0].view(-1)
    yv = orig_src_ind[:, :, 1].view(-1)
    xvnew = src_ind[:, :, 0].view(-1)
    yvnew = src_ind[:, :, 1].view(-1)

    # Set out of bounds indices to limits
    good_inds = (xvnew >= 0) * (yvnew >= 0) * (xvnew < im.size(2)) * (
        yvnew < im.size(3))

    batch_i = batch_i[good_inds].long()
    xv = xv[good_inds].long()
    yv = yv[good_inds].long()
    xvnew = xvnew[good_inds].long()
    yvnew = yvnew[good_inds].long()

    im_rot2 = torch.zeros_like(im)
    im_rot2[batch_i, :, xv, yv] = im[batch_i, :, xvnew, yvnew]
    return im_rot2


def position_encoding(featshape, n_dim=64):
    n_dim = n_dim
    n_freq = n_dim // 2
    div_term = torch.exp(
        torch.arange(0, n_freq, 2).float() * (-math.log(10000.0) / n_freq))
    height, width = featshape
    out = torch.zeros(n_dim, height, width)
    pos_w = torch.arange(-width // 2, width // 2 + height % 2).unsqueeze(1)
    pos_h = torch.arange(-height // 2, height // 2 + height % 2).unsqueeze(1)
    out[0:n_freq:2, :, :] = torch.sin(pos_h * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    out[1:n_freq:2, :, :] = torch.cos(pos_h * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    out[n_freq::2, :, :] = torch.sin(pos_w * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)
    out[n_freq + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)
    out = out.unsqueeze(0)
    return out


class PoolNonSequenceDimensions(nn.Module):
    """Docstring for PoolNonTemporalDimensions. """
    def __init__(self, pool_type='max', dim=2):
        """TODO: to be defined1. """
        nn.Module.__init__(self)
        self.pool_type = pool_type
        self.dim = dim

    def forward(self, x):
        """TODO: Docstring for forward.

        :x: TODO
        :returns: TODO

        """
        rem_dim = x.ndim - 1 - self.dim
        for _ in range(rem_dim):
            if self.pool_type == 'max':
                x = x.max(-1).values
            elif self.pool_type == 'avg':
                x = x.mean(-1)
        return x


def deconvBlock(input_nc,
                output_nc,
                bias=True,
                kernel_size=4,
                stride=2,
                padding=1,
                norm_layer=nn.BatchNorm3d,
                upscale_layer=nn.ConvTranspose3d,
                nl='lrelu'):
    layers = [
        upscale_layer(input_nc,
                      output_nc,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias)
    ]

    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    if nl == 'relu':
        layers += [nn.ReLU(True)]
    elif nl == 'lrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    else:
        raise NotImplementedError('NL layer {} is not implemented' % nl)
    return nn.Sequential(*layers)


def pool_predictions(output, nhoods):
    newoutput = [[o * 1 for o in output[0]]]
    for i in range(len(nhoods) - 1):
        npix = (nhoods[i] * np.array(output[i][0].shape[-2:]) /
                float(nhoods[i + 1])).astype(np.uint8)
        center = np.array(output[i + 1][0].shape[-2:]) // 2
        tmp_output = [
            F.interpolate(o, scale_factor=nhoods[i] / float(nhoods[i + 1]))
            for o in newoutput[-1]
        ]
        npix = np.array(tmp_output[0].shape[-2:])
        newoutput.append([o * 1 for o in output[i + 1]])
        for j in range(len(newoutput[-1])):
            newoutput[-1][j][:, :, center[0] - npix[0] // 2:center[0] +
                             npix[0] // 2 + npix[0] % 2, center[1] -
                             npix[1] // 2:center[1] + npix[1] // 2 +
                             npix[1] % 2] = tmp_output[j][:]
    return newoutput[-1:]


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x**2 + y**2)**0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu)**2 / (2 * sigma**2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma**2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x**2 + y**2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    sobel_2D = sobel_2D / np.sum(sobel_2D[:, 2])
    return sobel_2D


class EdgeAP(nn.Module):
    """Docstring for EdgePR. """
    def __init__(self):
        """TODO: to be defined1.

        :k: TODO

        """
        nn.Module.__init__(self)
        self.sobel = np.zeros((3, 3))
        self.sobel[1, :] = np.array([-0.5, 0, 0.5])
        self.sobel = nn.Parameter(
            torch.from_numpy(self.sobel).view(1, 1, *self.sobel.shape).float())

    def forward(self, pred, target, predictable=None, balance=False):
        """TODO: Docstring for forward.

        :x: TODO
        :returns: TODO

        """
        pred_edge, target_edge = edge_data(pred,
                                           target,
                                           self.sobel,
                                           predictable=predictable)
        pred_edge = pred_edge.cpu().detach()
        target_edge = (target_edge > 0).cpu().detach()

        output = pred_edge.numpy().flatten()
        target_edge = target_edge.numpy().flatten()
        if predictable is not None:
            predictable = predictable.cpu()
            target_edge = target_edge[predictable.view(-1) > 0]
            output = output[predictable.view(-1) > 0]
        sample_wt = np.ones_like(target_edge) / float(len(target_edge))
        if balance:
            n_pos = np.sum(target_edge > 0)
            n_neg = len(target_edge) - n_pos
            sample_wt[target_edge > 0] = 0.5 / (float(n_pos) + 1e-10)
            sample_wt[target_edge == 0] = 0.5 / (float(n_neg) + 1e-10)

        ap = sklearn.metrics.average_precision_score(target_edge,
                                                     output,
                                                     sample_weight=sample_wt)
        return ap


class EdgePR(nn.Module):
    """Docstring for EdgePR. """
    def __init__(self, k=3):
        """TODO: to be defined1.

        :k: TODO

        """
        nn.Module.__init__(self)
        self._k = k
        self.sobel = np.zeros((3, 3))
        self.sobel[1, :] = np.array([-0.5, 0, 0.5])
        self.sobel = nn.Parameter(
            torch.from_numpy(self.sobel).view(1, 1, *self.sobel.shape).float())

    def forward(self, pred, target):
        """TODO: Docstring for forward.

        :x: TODO
        :returns: TODO

        """
        pred_edge, target_edge = edge_data(pred, target, self.sobel)
        #pred_edge = (pred_edge > 0.25)
        #target_edge = (target_edge > 0.25)
        precision = (pred_edge * target_edge).sum().float() / (
            (pred_edge > 0).sum().float() + 1e-10)
        recall = (pred_edge * target_edge).sum().float() / (
            (target_edge > 0).sum().float() + 1e-10)
        return precision, recall


def edge_data(pred, target, sobel, predictable=None):
    target_edge_x = F.conv2d(target.unsqueeze(1).float(), sobel, padding=1)
    target_edge_y = F.conv2d(target.unsqueeze(1).float(),
                             sobel.permute(0, 1, 3, 2),
                             padding=1)
    target_edge = (target_edge_x**2 + target_edge_y**2)**0.5

    pred_edge_x = F.conv2d(pred.unsqueeze(1).float(), sobel, padding=1)
    pred_edge_y = F.conv2d(pred.unsqueeze(1).float(),
                           sobel.permute(0, 1, 3, 2),
                           padding=1)
    pred_edge = (pred_edge_x**2 + pred_edge_y**2)**0.5
    if predictable is not None:
        predictable_edge_x = F.conv2d(predictable.unsqueeze(1).float(),
                                      sobel,
                                      padding=1)
        predictable_edge_y = F.conv2d(predictable.unsqueeze(1).float(),
                                      sobel.permute(0, 1, 3, 2),
                                      padding=1)
        predictable_edge = ((predictable_edge_x**2 + predictable_edge_y**2)**
                            0.5) > 0
        bad_edges = ((~(predictable > 0).unsqueeze(1)) + predictable_edge)

        target_edge[bad_edges] = target_edge.min()
        pred_edge[bad_edges] = pred_edge.min()
    return pred_edge, target_edge
