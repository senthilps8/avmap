import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as korntransforms

import numpy as np
import models.utils as model_utils
import math
import models.unet_parts as unet_parts
import models.transformer_parts as transformer_parts


class BaseOccupancy2DPredictor(nn.Module):
    def __init__(self, cfg):
        """TODO: to be defined1.

        :cfg: TODO

        """
        nn.Module.__init__(self)

        self.cfg = cfg
        self.pool_steps = self.cfg.pool_steps
        self.output_padding = self.cfg.output_padding
        self.step_size = self.cfg.step_size
        self.transformer_use_padding_mask = self.cfg.transformer_use_padding_mask

        self.rgb_feat, rgb_ndim = self.create_rgb_feature_extractor(
            self.cfg.rgb_model)
        self.audio_feat, audio_ndim = self.create_audio_feature_extractor(
            self.cfg.audio_model)

        self.padded_enc_in_scale = np.array(self.cfg.enc_in_scale)
        self.dec_out_scale = np.array(self.cfg.decoder_model.output_gridsize)
        predictable_region = torch.ones(
            (1, 1, self.dec_out_scale[0], self.dec_out_scale[1]))

        # Input sizes to encoders will increase if
        # we want to align features
        # We will use this to pad
        for si in range(len(self.padded_enc_in_scale)):
            self.padded_enc_in_scale[si] += 2 * int(
                np.ceil(
                    np.sqrt(2) * self.output_padding *
                    self.padded_enc_in_scale[si] /
                    float(self.cfg.decoder_model.output_gridsize[si])))

        # Change output gridsize so that the model
        # outputs directly in the translated+rotated outputs
        # since the features are already aligned
        self.dec_out_scale = [
            sc + 2 * self.output_padding
            for sc in self.cfg.decoder_model.output_gridsize
        ]
        predictable_region = F.pad(predictable_region,
                                   (self.output_padding, self.output_padding,
                                    self.output_padding, self.output_padding))

        self.register_buffer('predictable_region', predictable_region)
        self.dec_in_scale = self.padded_enc_in_scale

        # Model output is the samesize always
        # Only decoder output size changes based on whether we alignn features
        # or align outputs
        self.model_out_scale = np.array(self.cfg.decoder_model.output_gridsize)
        self.model_out_scale = [
            sc + 2 * self.output_padding
            for sc in self.cfg.decoder_model.output_gridsize
        ]
        print('Feature Output Scale', self.cfg.enc_in_scale)
        print('Encoder Input Scale', self.padded_enc_in_scale)
        print('Decoder Input size', self.dec_in_scale)
        print('Decoder Output size', self.dec_out_scale)
        print('Model Output size', self.model_out_scale)

        # Input num channels to encoders will increase if
        # we want to concat position encoding
        rgb_ndim = rgb_ndim + 64
        audio_ndim = audio_ndim + 64

        self.rgb_enc, rgb_enc_scale = self.create_rgb_encoder(
            self.cfg.rgb_model, rgb_ndim)
        self.audio_enc, aud_enc_scale = self.create_audio_encoder(
            self.cfg.audio_model, audio_ndim)
        num_inputs = sum([
            int(self.cfg.rgb_model.use_model) *
            self.cfg.rgb_model.encoder.channel_factor,
            int(self.cfg.audio_model.use_model) *
            self.cfg.audio_model.encoder.channel_factor,
        ])
        if self.cfg.rgb_model.use_model:
            enc_out_scale = rgb_enc_scale
        elif self.cfg.audio_model.use_model:
            enc_out_scale = aud_enc_scale

        self.decoder = self.create_occupancy_decoder(
            self.cfg.decoder_model,
            nsf=num_inputs,
            input_scale=enc_out_scale,
        )
        print(self)

    def create_rgb_feature_extractor(self, cfg):
        if not cfg.use_model:
            return nn.Sequential(), 0
        model = unet_parts.ResNetRGBEncoderByProjection(
            pretrained=cfg.pretrained,
            out_scale=tuple(cfg.encoder.in_scale),
            resnet_type=cfg.resnet_type,
            norm=cfg.norm)
        return model, model.n_channels_out

    def create_audio_feature_extractor(self, cfg):
        if not cfg.use_model:
            return nn.Sequential(), 0
        relu = nn.ReLU(inplace=True)

        if cfg.norm == 'batchnorm':
            normlayer = nn.BatchNorm1d
            fcnormlayer = nn.BatchNorm1d
        elif cfg.norm == 'instancenorm':
            normlayer = nn.InstanceNorm1d

            def fcnormlayer(nchannels):
                return nn.GroupNorm(1, nchannels)

        first_stride = 1
        kernel_size = 51
        pooling_size = 4

        audio_conv1 = nn.Conv1d(9,
                                16,
                                kernel_size=(kernel_size),
                                stride=first_stride)
        audio_bn1 = normlayer(16)
        audio_pool1 = nn.MaxPool1d(kernel_size=(10), stride=(pooling_size))
        audio_conv2 = nn.Conv1d(16, 32, kernel_size=(kernel_size), stride=1)
        audio_bn2 = normlayer(32)
        audio_pool2 = nn.MaxPool1d(kernel_size=(10), stride=(pooling_size))
        audio_conv3 = nn.Conv1d(32, 64, kernel_size=(kernel_size), stride=1)
        audio_bn3 = normlayer(64)
        audio_pool3 = nn.MaxPool1d(kernel_size=(10), stride=(pooling_size))
        audio_conv4 = nn.Conv1d(64, 64, kernel_size=(kernel_size), stride=1)
        audio_bn4 = normlayer(64)

        pool_seq = model_utils.PoolNonSequenceDimensions(dim=1)
        fc1 = nn.Linear(64, cfg.feature.feature_dim)
        fc1_bn = fcnormlayer(cfg.feature.feature_dim)
        upsampler = unet_parts.Scaler(cfg.feature.feature_dim,
                                      tuple(cfg.encoder.in_scale),
                                      norm=cfg.norm)

        layers = [
            audio_conv1, audio_bn1, relu, audio_pool1, audio_conv2, audio_bn2,
            relu, audio_pool2, audio_conv3, audio_bn3, relu, audio_pool3,
            audio_conv4, audio_bn4, relu, pool_seq, fc1, fc1_bn, relu,
            upsampler
        ]
        if cfg.input_length > 0:
            layers = [nn.AdaptiveMaxPool1d(cfg.input_length)] + layers
        return nn.Sequential(*layers), cfg.feature.feature_dim

    def create_rgb_encoder(self, cfg, nch_in):
        if not cfg.use_model:
            return nn.Sequential(), np.array((0, 0))
        modfunc = transformer_parts.TransformerSpatialUNetEncoder
        model = modfunc(
            nch_in,
            nsf=cfg.encoder.channel_factor,
            n_downscale=cfg.encoder.n_downscale,
            norm=cfg.norm,
            nhead=cfg.encoder.transformer_nhead,
            prenorm=cfg.encoder.transformer_prenorm,
            dropout=cfg.dropout,
        )
        output_scale = np.array(self.padded_enc_in_scale) // 2**int(
            cfg.encoder.n_downscale)
        return model, output_scale

    def create_audio_encoder(self, cfg, nch_in):
        if not cfg.use_model:
            return nn.Sequential(), np.array((0, 0))
        modfunc = transformer_parts.TransformerSpatialUNetEncoder
        model = modfunc(nch_in,
                        nsf=cfg.encoder.channel_factor,
                        n_downscale=cfg.encoder.n_downscale,
                        norm=cfg.norm,
                        nhead=cfg.encoder.transformer_nhead,
                        dropout=cfg.dropout,
                        prenorm=cfg.encoder.transformer_prenorm)
        output_scale = np.array(self.padded_enc_in_scale) // 2**int(
            cfg.encoder.n_downscale)
        return model, output_scale

    def create_occupancy_decoder(self, cfg, nsf, input_scale):
        modfunc = transformer_parts.TransformerSpatialUNetDecoder
        model = modfunc(nsf=nsf,
                        in_scale=input_scale,
                        out_scale=np.array(self.dec_out_scale),
                        n_downscale=cfg.n_downscale,
                        n_upscale=int(
                            np.ceil(
                                math.log(
                                    np.amax(self.dec_out_scale) /
                                    np.amin(input_scale), 2))),
                        norm=cfg.norm,
                        nhead=cfg.transformer_nhead,
                        prenorm=cfg.transformer_prenorm,
                        dropout=cfg.dropout,
                        n_classes=cfg.n_classes)
        return model

    def merge_inputs(self, inputs):
        out = []
        num_inputs = len(inputs)
        num_samples = inputs[0][0].shape[0]
        drop_weights = torch.from_numpy(
            np.ones((num_samples, num_inputs, 1, 1,
                     1)).astype(np.float32)).to(inputs[0][0].device)
        for k in range(len(inputs[0])):
            out.append(
                torch.cat([
                    x[k] * drop_weights[:, xi] for xi, x in enumerate(inputs)
                ], 1))
        return out

    def forward(self, x):
        raise NotImplementedError


class SequenceOccupancySemanticsPredictor(BaseOccupancy2DPredictor):
    def __init__(self, cfg):
        """TODO: to be defined1. """
        BaseOccupancy2DPredictor.__init__(self, cfg)
        assert (cfg.decoder_model.out_nscales == 1)
        self.n_steps = self.cfg.n_steps
        self.register_buffer(
            'position_feat',
            model_utils.position_encoding(np.array(self.dec_in_scale), 64))

    def translate_and_rotate(self, x, paths):
        paths = paths.clone().view(-1, 4)
        x = x.float()
        rotation = (paths[:, -1].float() + 360) % 360
        translation = torch.cat([paths[:, 1:2], paths[:, :1]], 1).float()
        x_rot = korntransforms.rotate(x,
                                      angle=rotation,
                                      center=None,
                                      mode='nearest')
        x = korntransforms.translate(x_rot, translation=translation)
        return x

    def pool_predictions(self, x, paths, clean_predictions=False):
        if clean_predictions:
            predictable_regions = self.translate_and_rotate(
                self.predictable_region.repeat(x.shape[0], 1, 1, 1), paths)
            # Minor bug: need to set to -inf not zero, does not affect results
            x = x * predictable_regions
        start_inds = np.cumsum([0] + [len(p) for p in paths])
        x = torch.cat([
            torch.max(x[start_inds[i]:start_inds[i] + len(paths[i])],
                      0,
                      keepdim=True).values for i in range(len(paths))
        ], 0)
        return x

    def pad_translate_rotate_features(self, feat, enc_in_scale,
                                      output_relpath):
        relpath = output_relpath.clone()
        relpath[:, :, 0] *= (feat.shape[-2] /
                             float(self.cfg.decoder_model.output_gridsize[0]))
        relpath[:, :, 1] *= (feat.shape[-1] /
                             float(self.cfg.decoder_model.output_gridsize[1]))

        feat = F.pad(
            feat,
            [int(np.ceil(np.amax(
                (enc_in_scale - feat.shape[-2:]) / 2.0)))] * 4)
        if self.cfg.position_encoding:
            feat = torch.cat(
                [feat,
                 self.position_feat.expand(feat.shape[0], -1, -1, -1)], 1)

        alignedfeat = self.translate_and_rotate(feat, relpath)
        return alignedfeat

    def modality_forward(self, data, relpath, model):
        # convert each to sum(N_step)*B X 3 X H X W before processing
        data = data.view(-1, *data.shape[2:])
        data_x = model(data)
        data_x = self.pad_translate_rotate_features(
            data_x, np.array(self.padded_enc_in_scale), relpath)
        data_feat = data_x
        if self.cfg.position_encoding:
            data_feat = data_feat[:, :-self.position_feat.shape[1]]
        return data_x, data_feat

    def forward(self, rgb=None, audio=None, relpath=None, padding=0):

        # Step1: Extract Independent Features and Align
        out_feat = {}
        if self.cfg.rgb_model.use_model:
            rgb_x, rgb_feat = self.modality_forward(rgb, relpath,
                                                    self.rgb_feat)
            out_feat['rgb'] = rgb_feat
        if self.cfg.audio_model.use_model:
            audio_x, audio_feat = self.modality_forward(
                audio, relpath, self.audio_feat)
            out_feat['audio'] = audio_feat

        mask = None
        if self.transformer_use_padding_mask:
            feat_aligned = next(iter(out_feat.values()))
            mask = (feat_aligned == 0).all(1)

        # Step2: Self-Attention based UNet Encoder
        encoded_feats = []
        if self.cfg.rgb_model.use_model:
            rgb_feat_levels = self.rgb_enc(rgb_x, relpath, mask_x=mask)
            encoded_feats.append(rgb_feat_levels)
        if self.cfg.audio_model.use_model:
            audio_feat_levels = self.audio_enc(audio_x, relpath, mask_x=mask)
            encoded_feats.append(audio_feat_levels)
        encoded_feats = self.merge_inputs(encoded_feats)

        # Step3: Self-Attention based UNet Decoder
        allpred = self.decoder(encoded_feats, relpath, mask)
        # Split into interior and room predictions
        occ_ant = allpred[:, :2]
        # Quick Hack to convert to sigmoid when softmax is applied
        # Need to remove and switch to a single prediction + sigmoid
        occ_ant[:, 0] = 0.0
        cls_ant = allpred[:, 2:]

        if self.pool_steps:
            # Only for evaluation (in train: loss on unpooled predictions)
            occ_ant = self.pool_predictions(occ_ant,
                                            relpath,
                                            clean_predictions=True)
            cls_ant = self.pool_predictions(cls_ant,
                                            relpath,
                                            clean_predictions=True)
        return occ_ant, cls_ant, out_feat
