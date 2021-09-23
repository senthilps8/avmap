import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleWeightedCrossEntropy(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self._cfg = cfg

    def forward(self, output, target, meta=None, return_spatial=False):
        loss = F.cross_entropy(output, target, reduction='none')
        # If GT has padding, ignore the points that are not predictable
        if meta is not None and 'predictable_target' in meta:
            pred_index = meta['predictable_target'].long()
        else:
            pred_index = torch.ones_like(target).long()

        spatial_loss = loss
        if self._cfg.loss.binary_balanced:
            loss = 0.5 * loss[pred_index * target > 0].mean() + 0.5 * loss[
                pred_index * (1 - target) > 0].mean()
        else:
            loss = loss[pred_index > 0].mean()
        if return_spatial:
            return loss, spatial_loss
        return loss


class NonZeroWeightedCrossEntropy(SimpleWeightedCrossEntropy):
    def __init__(self, cfg):
        SimpleWeightedCrossEntropy.__init__(self, cfg)

    def forward(self, output, target, meta=None, return_spatial=False):
        if meta is not None and 'predictable_target' in meta:
            pred_index = meta['predictable_target']
        else:
            pred_index = torch.ones_like(target)

        pred_index = pred_index * (target > 0).long()
        loss = F.cross_entropy(output, (target - 1) % target.max(),
                               reduction='none')
        spatial_loss = loss

        loss = loss[pred_index > 0].mean() if torch.sum(
            pred_index > 0) else torch.tensor(0.0).to(pred_index.device)
        if return_spatial:
            return loss, spatial_loss
        return loss
