import numpy as np
import models.unet_parts as unet_parts
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class Batch3DtoSeq1D(nn.Module):
    """Docstring for Batch3DtoSeq1D. """
    def __init__(self):
        """TODO: to be defined. """
        nn.Module.__init__(self)

    def forward(self, x, seqlens, mask_x=None):
        """TODO: Docstring for forward.

        :x: data with dimensions as sum(seqlens) X C X H X W
        :seqlens: 1D tensor of sequence lengths with dim N
        :returns:
        :   x: 3D Tensor with dims max(seqlens) X NHW X C, dim 0 is zero-padded
        :   mask: 2D Bool Tensor with dims max(seqlens) X NHW (True: Padding)

        """
        # Warning: this function assumes all sequences in the batch
        # are of same length
        maxseqlen = seqlens[0]
        # seqlen_cumsum = torch.cumsum(F.pad(seqlens, (1, 0)), 0)

        # pad, and cat
        sequences = x.view(-1, maxseqlen, x.shape[1], x.shape[2], x.shape[3])
        if mask_x is None:
            mask_sequences = torch.zeros_like(sequences[:, :, 0])
        else:
            mask_sequences = mask_x.view(-1, maxseqlen, mask_x.shape[1],
                                         mask_x.shape[2])
        x = sequences
        mask = mask_sequences

        # merge N, H,W into single dimension
        #   - each i,j position is considered an independent sequence
        #   - may need to add position encoding here
        x = x.permute(1, 0, 3, 4, 2).contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1])
        mask = mask.permute(1, 0, 2, 3).contiguous()
        mask = mask.view(mask.shape[0], -1) > 0
        mask = mask.t()
        return x, mask


class Seq1DtoBatch3D(nn.Module):
    """Docstring for Batch3DtoSeq1D. """
    def __init__(self):
        """TODO: to be defined. """
        nn.Module.__init__(self)

    def forward(self, x, seqlens, out_shape):
        """TODO: Docstring for forward.

        :x: data with dimensions as sum(seqlens) X C X H X W
        :seqlens: 1D tensor of sequence lengths with dim N
        :returns:
        :   x: 3D Tensor with dims max(seqlens) X NHW X C, dim 0 is zero-padded
        :   mask: 2D Bool Tensor with dims max(seqlens) X NHW (True: Padding)

        """
        # First convert back to max(seqlens) X N X H X W X C
        x = x.view(-1, len(seqlens), out_shape[-2], out_shape[-1], x.shape[-1])

        # Now max(seqlens) X N X C X H X W
        warnings.warn('Assuming all seq eq length')
        #x = x.permute(1, 0, 4, 2, 3)
        x = x.permute(1, 0, 4, 2, 3)
        # Now sum(seqlens) X C X H X W
        x = x.contiguous().view(-1, *x.shape[2:])
        #x = torch.cat([x[:seqlens[i], i] for i in range(len(seqlens))], 0)
        return x


class SpatialTransformerEncoderLayer(nn.Module):
    """Docstring for SpatialTransformerEncoderLayer. """
    def __init__(self, n_channels, nhead, dropout=0.0, prenorm=False):
        """TODO: to be defined1.

        :n_channels: TODO
        :nhead: TODO
        :dropout: TODO

        """
        nn.Module.__init__(self)

        self._n_channels = n_channels
        self._nhead = nhead
        self._dropout = dropout
        self._prenorm = prenorm
        self.convert3dto1d = Batch3DtoSeq1D()
        self.convert1dto3d = Seq1DtoBatch3D()
        self.mheadattn = nn.MultiheadAttention(
            n_channels,
            nhead,
            dropout=dropout,
        )
        self.norm = nn.GroupNorm(1, n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, seqlens, mask_src=None):
        """TODO: Docstring for forward.

        :src: TODO
        :src_mask: TODO
        :src_key_padding_mask: TODO
        :returns: TODO

        """
        srcin = src
        orig_shape = src.shape
        if self._prenorm:
            srcin = self.norm(src)
        srcin, padmask = self.convert3dto1d(srcin, seqlens, mask_src)
        out_src = torch.zeros_like(srcin)
        posind = torch.where(~padmask.all(1))[0]
        srcin = srcin[:, posind]
        padmask = padmask[posind]
        src2 = self.mheadattn(srcin, srcin, srcin, key_padding_mask=padmask)[0]
        out_src[:, posind] = out_src[:, posind] + src2
        src2 = out_src
        src2 = self.dropout(src2)
        src = src + self.convert1dto3d(src2, seqlens, orig_shape)
        if not self._prenorm:
            src = self.norm(src)
        return src


# =============== Component modules ====================


class Pass(nn.Module):
    """Docstring for Pass. """
    def __init__(self):
        """TODO: to be defined1. """
        nn.Module.__init__(self)

    def forward(self, *args):
        """TODO: Docstring for forward.

        :*args: TODO
        :returns: TODO

        """
        return args[0]


class TransformerSpatialUNetEncoder(unet_parts.UNetEncoder):
    def __init__(self,
                 n_channels,
                 nsf=16,
                 n_downscale=4,
                 norm='batchnorm',
                 nhead=4,
                 dropout=0.0,
                 prenorm=False):
        super().__init__(n_channels=n_channels,
                         nsf=nsf,
                         n_downscale=n_downscale,
                         norm=norm)
        self.mheadattn_in = Pass()

        self.mheadattn_mods = nn.ModuleList([
            SpatialTransformerEncoderLayer(nsf * (2**min(i, 3)),
                                           min(nhead, nsf * (2**min(i, 3))),
                                           dropout=dropout,
                                           prenorm=prenorm)
            for i in range(n_downscale)
        ])

    def forward(self, x, paths, mask_x=None):
        seqlens = torch.sum(paths[:, :, 0] > -1000, 1)
        x = self.mheadattn_in(x, seqlens, mask_x)
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        out_x = [x1]
        for mi, mod in enumerate(self.down_mods):
            if mask_x is not None:
                mask_x = F.interpolate(
                    mask_x.unsqueeze(1).float(),
                    out_x[-1].shape[-2:]).squeeze(1) > 0
            x_attn = self.mheadattn_mods[mi](out_x[-1], seqlens, mask_x)
            out_x.append(mod(x_attn))

        return out_x  # {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}


class TransformerSpatialUNetDecoder(unet_parts.UNetDecoder):
    def __init__(self,
                 n_classes=2,
                 nsf=16,
                 n_downscale=4,
                 n_upscale=4,
                 in_scale=np.array([32, 32]),
                 out_scale=np.array([64, 64]),
                 norm='batchnorm',
                 nhead=4,
                 dropout=0.0,
                 prenorm=False):
        super().__init__(n_classes=n_classes,
                         nsf=nsf,
                         n_downscale=n_downscale,
                         n_upscale=n_upscale,
                         in_scale=in_scale,
                         out_scale=out_scale,
                         norm=norm)
        mheadattn_mods = [
            SpatialTransformerEncoderLayer(nsf * (2**np.clip(i + 1, 0, 3)),
                                           nhead,
                                           dropout=dropout,
                                           prenorm=prenorm)
            for i in range(n_downscale - 1, -1, -1)
        ]
        mheadattn_mods = mheadattn_mods + [
            SpatialTransformerEncoderLayer(nsf, nhead, dropout=dropout)
            for i in range(n_upscale - n_downscale)
        ]
        self.mheadattn_mods = nn.ModuleList(mheadattn_mods)

    def forward(self, xin, paths, mask_x=None):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        seqlens = torch.sum(paths[:, :, 0] > -1000, 1)

        x = xin.pop()
        for mi, mod in enumerate(self.up_mods):
            if mask_x is not None:
                mask_x = F.interpolate(
                    mask_x.unsqueeze(1).float(), x.shape[-2:]).squeeze(1) > 0
            x = self.mheadattn_mods[mi](x, seqlens, mask_x)
            if len(xin):
                x = mod(x, xin.pop())
            else:
                x = mod(x)

        x = self.outc(x)  # (bs, n_classes, ..., ...)
        x = x[:, :, (x.shape[-2] // 2) -
              (self.out_scale[0] // 2):(x.shape[-2] // 2) +
              (self.out_scale[0] // 2) +
              (self.out_scale[0] % 2), (x.shape[-1] // 2) -
              (self.out_scale[1] // 2):(x.shape[-1] // 2) +
              (self.out_scale[1] // 2) + (self.out_scale[1] % 2)]
        return x
