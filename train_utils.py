import torch
import numpy as np
from pathlib import Path
import hydra.utils as hydra_utils
import os


def save_checkpoint(state, fname_fmt='checkpoint_{:04d}.pth'):
    torch.save(state, fname_fmt.format(state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', tbnames=[]):
        self.name = name
        self.fmt = fmt
        self.tbnames = tbnames
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            val = val.flatten()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if isinstance(self.val, np.ndarray):
            fmtstr = '{name}: \n\t    {val' + self.fmt + \
                '} \n\t   ({avg' + self.fmt + '})'
        else:
            fmtstr = '{name}: \n\t    {val' + \
                self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", tbwriter=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tbwriter = tbwriter

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\n\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def tbwrite(self, batch):
        if self.tbwriter is None:
            return
        scalar_dict = self.tb_scalar_dict()
        for k, v in scalar_dict.items():
            self.tbwriter.add_scalar(k, v, batch)

    def tb_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            if not isinstance(val, np.ndarray):
                val = [val]
            if len(meter.tbnames) == 0:
                if len(val) > 1:
                    meter.tbnames = [
                        meter.name + '_{}'.format(i) for i in range(len(val))
                    ]
                else:
                    meter.tbnames = [meter.name]
            for i in range(len(val)):
                tag = meter.tbnames[i]
                sclrval = val[i]
                out[tag] = sclrval
        return out


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sort(np.array(list(cfg.steps)))
    n_crossed = np.searchsorted(steps, epoch, side='right')
    lr = cfg.lr * (0.1**n_crossed)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # Written by: Achal Dave
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
