from train_utils import (save_checkpoint, AverageMeter, ProgressMeter,
                         adjust_learning_rate,
                         update_pythonpath_relative_hydra)
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torch.optim
import torch.nn.parallel
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
import torch.cuda
import torch
import submitit
import hydra.utils as hydra_utils
import hydra
import json
from collections import defaultdict
import os
import sys
import copy
import time
import models
import models.avmap
import models.losses
import models.utils
import data.utils as data_utils
import numpy as np
import pdb
np.set_printoptions(precision=3)
os.environ['OMP_NUM_THREADS'] = '2'


class Worker:
    def __init__(self, train_func, val_func):
        self.train_func = train_func
        self.val_func = val_func

    def __call__(self, origargs):
        args = copy.deepcopy(origargs)
        print(origargs.pretty())
        np.set_printoptions(precision=3)
        import torch.backends.cudnn
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('Experiment name: {}'.format(args.logging.name))
        print('Suffix: {}'.format(args.logging.suffix))
        args.environment.dist_url = f'tcp://localhost:{args.environment.port}'
        print('Using url {}'.format(args.environment.dist_url))
        args.environment.distributed = args.environment.world_size > 1 or args.environment.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        print('Found {} gpus'.format(ngpus_per_node))
        if args.environment.multiprocessing_distributed:
            args.environment.world_size = ngpus_per_node * args.environment.world_size
            mp.spawn(main_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            self, *args, **kwargs)  # submits to requeuing


def main_worker(gpu, ngpus_per_node, args):
    np.set_printoptions(precision=3)
    import models
    import models.losses
    import data
    import models.avmap
    import data.rgba_dataset
    import torch.backends.cudnn as cudnn
    import builtins
    import torch.distributed as dist
    cudnn.benchmark = True
    args.environment.gpu = gpu

    # suppress printing if not master
    if args.environment.multiprocessing_distributed and args.environment.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.environment.gpu is not None:
        print("Use GPU: {} for training".format(args.environment.gpu))

    if args.environment.distributed:
        if args.environment.multiprocessing_distributed:
            args.environment.rank = args.environment.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.environment.dist_backend,
                                init_method=args.environment.dist_url,
                                world_size=args.environment.world_size,
                                rank=args.environment.rank)

    writer = None
    if args.logging.log_tb:
        logdir = os.path.join(args.logging.tb_dir,
                              args.logging.name + args.logging.suffix)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    os.makedirs(os.path.join(args.logging.ckpt_dir, args.logging.name),
                exist_ok=True)
    ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                              'checkpoint_{:04d}.pth')

    # Create Datasets and set padding, step size and n_classes
    dataset_func = data.rgba_dataset.getRGBAmbisonicsAreaDataset
    train_dataset, all_val_datasets = dataset_func(args.data)
    n_semantic_categories = train_dataset.n_categories
    args.model.output_padding = train_dataset.padding
    args.model.step_size = int(train_dataset.step_size)
    args.model.decoder_model.n_classes = 2 + n_semantic_categories
    if args.environment.evaluate_path != '':
        evaluate_path(args,
                      all_val_datasets,
                      args.environment.evaluate_path,
                      writer=writer)
        sys.exit(0)

    # Create Model and losses
    model = models.avmap.SequenceOccupancySemanticsPredictor(args.model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = models.losses.SimpleWeightedCrossEntropy(args).cuda(
        args.environment.gpu)
    cls_criterion = models.losses.NonZeroWeightedCrossEntropy(args).cuda(
        args.environment.gpu)
    print(model)

    if args.environment.distributed:
        torch.cuda.set_device(args.environment.gpu)
        model.cuda(args.environment.gpu)
        args.optim.batch_size = int(args.optim.batch_size / ngpus_per_node)
        args.environment.workers = int(
            (args.environment.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.environment.gpu])
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = torch.optim.SGD(model.parameters(),
                                args.optim.lr,
                                momentum=args.optim.momentum,
                                weight_decay=args.optim.weight_decay)

    start_epoch = 0
    start_iter = 0
    for i in range(args.optim.epochs, -1, -1):
        if os.path.exists(ckpt_fname.format(i)):
            print('loading file {}'.format(ckpt_fname.format(i)))
            checkpoint = torch.load(ckpt_fname.format(i))
            start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iteration']
            if 'module.predictable_region' in checkpoint['state_dict']:
                del checkpoint['state_dict']['module.predictable_region']
            if 'module.position_feat' in checkpoint['state_dict']:
                del checkpoint['state_dict']['module.position_feat']
            paramnames = list(checkpoint['state_dict'].keys())
            for k in paramnames:
                if 'decoders.' in k:
                    checkpoint['state_dict'][k.replace(
                        'decoders.0.', '')] = checkpoint['state_dict'][k]
                    del checkpoint['state_dict'][k]
            paramnames = list(checkpoint['state_dict'].keys())
            for k in paramnames:
                if 'outc.0' in k:
                    checkpoint['state_dict'][k.replace(
                        'outc.0', 'outc')] = checkpoint['state_dict'][k]
                    del checkpoint['state_dict'][k]

            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(msg)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint, starting at Epoch {}".format(
                start_epoch))
            loaded = True
            break
        if i == 0:
            print("=> no checkpoint found")

    if args.environment.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # Train model
    for epoch in range(start_epoch, args.optim.epochs):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.optim.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.environment.workers,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=data_utils.worker_init_fn)
        adjust_learning_rate(optimizer, epoch, args.optim)
        # train for one epoch
        train_seq(train_loader, model, criterion, cls_criterion, optimizer,
                  epoch, start_iter, args, writer)

        if args.environment.gpu == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'iteration': 0,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_fname.format(epoch))
            start_iter = 0


def train_seq(train_loader,
              model,
              criterion,
              cls_criterion,
              optimizer,
              epoch,
              start_iter,
              args,
              writer=None):
    out_nhoods = [args.data.out_nhood]
    out_scales = [list(np.array(args.data.output_gridsize))]
    out_scales = ['X'.join([str(d) for d in out_scales[0]])]
    tbsuffixes = ['_{}_{}'.format(out_nhoods[0], out_scales[0])]
    batch_time = AverageMeter('Time', ':04.2f', tbnames=['train/time'])
    data_time = AverageMeter('Data', ':04.2f', tbnames=['train/datatime'])
    losses = AverageMeter('Loss', '', tbnames=['train/loss' + tbsuffixes[0]])
    clslosses = AverageMeter('ClsLoss',
                             ':04.2f',
                             tbnames=['train/clsloss' + tbsuffixes[0]])
    top1 = AverageMeter('Acc@1',
                        ':04.2f',
                        tbnames=['train/acc' + tbsuffixes[0]])
    ious = AverageMeter('IOU', ':04.2f', tbnames=['train/iou' + tbsuffixes[0]])
    aps = AverageMeter('AP', ':04.2f', tbnames=['train/ap' + tbsuffixes[0]])
    edge_aps = AverageMeter('E-AP',
                            ':04.2f',
                            tbnames=['train/edgeap' + tbsuffixes[0]])

    progress = ProgressMeter(
        len(train_loader) + start_iter,
        [batch_time, data_time, losses, clslosses, top1, ious, aps, edge_aps],
        prefix="Epoch: [{}]".format(epoch),
        tbwriter=writer)
    compute_edge_ap = models.utils.EdgeAP()
    compute_edge_ap = compute_edge_ap.cuda(args.environment.gpu)
    ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                              'checkpoint_{:04d}.pth')

    model.train()
    end = time.time()
    for batchi, (data, target, meta) in enumerate(train_loader):
        i = batchi + start_iter
        audio, rgb, relpath, target, meta = data_utils.prepare_batch(
            data, target, meta)

        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        (output, cls_output,
         finalpad) = model(rgb=rgb,
                           audio=audio,
                           relpath=relpath,
                           padding=meta['sequence_padding'][0].item())
        loss = criterion(output, target, meta)
        cls_loss = cls_criterion(cls_output, meta['semantic_target'], meta)

        # Compute all the metrics and baselines
        losses.update(loss.item(), audio.size(0))
        clslosses.update(cls_loss.item(), audio.size(0))
        if (batchi + 1) % args.logging.train_eval_freq == 0:
            with torch.no_grad():
                prob = F.softmax(output, 1)
                acc1 = data_utils.accuracy(output.permute(0, 2, 3,
                                                          1).contiguous().view(
                                                              -1, 2),
                                           target.view(-1),
                                           topk=(1, ))[0][0].item()
                pred = (output[:, 0] < output[:, 1]
                        ).long().detach() * meta['predictable_target']
                pred_scores = prob[:, 1].detach()
                pred_scores[meta['predictable_target'] ==
                            0] = pred_scores.min()
                iou = data_utils.IOU(pred, target).item()
                ap = data_utils.AP(pred_scores.cpu(), target.cpu())
                edge_ap = compute_edge_ap(pred_scores, target,
                                          meta['predictable_target'])
                # Update progress meter
                top1.update(acc1, audio.size(0))
                ious.update(iou, audio.size(0))
                aps.update(ap, audio.size(0))
                edge_aps.update(edge_ap, audio.size(0))
                progress.tbwrite(epoch * len(train_loader.dataset) //
                                 args.optim.batch_size + i)

        # Compute gradient and do SGD step
        scaled_loss = (loss + cls_loss) / float(args.optim.n_batches_update)
        scaled_loss.backward()
        if args.optim.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(),
                                     args.optim.max_grad_norm)
        if (batchi + 1) % args.optim.n_batches_update == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.logging.print_freq == 0:
            progress.display(i)
    optimizer.zero_grad()


def evaluate_path(args, all_val_datasets, ckpt_path, writer=None):
    res_fname = ckpt_path.replace(
        '.pth', '') + '{:04d}steps_res' + args.logging.suffix + '.pth'

    start_epoch = 0
    start_iter = 0
    loaded = False
    if not os.path.exists(ckpt_path):
        return
    for val_nsteps, val_dataset in all_val_datasets.items():
        print('Evaluation validation with {} steps'.format(val_nsteps))
        args.model.n_steps = val_nsteps
        args.model.output_padding = val_dataset.padding
        args.model.step_size = int(val_dataset.step_size)
        model = models.avmap.SequenceOccupancySemanticsPredictor(args.model)
        criterion = models.losses.SimpleWeightedCrossEntropy(args).cuda(
            args.environment.gpu)
        cls_criterion = models.losses.NonZeroWeightedCrossEntropy(args).cuda(
            args.environment.gpu)
        model = torch.nn.DataParallel(model).cuda(args.environment.gpu)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.optim.val_batch_size,
            shuffle=False,
            num_workers=args.environment.workers,
            pin_memory=True,
            sampler=None)

        if os.path.exists(res_fname.format(val_nsteps)):
            print('Result exists...')
            continue

        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        del checkpoint['state_dict']['module.predictable_region']
        del checkpoint['state_dict']['module.position_feat']
        paramnames = list(checkpoint['state_dict'].keys())
        for k in paramnames:
            if 'decoders.' in k:
                checkpoint['state_dict'][k.replace(
                    'decoders.0.', '')] = checkpoint['state_dict'][k]
                del checkpoint['state_dict'][k]
        paramnames = list(checkpoint['state_dict'].keys())
        for k in paramnames:
            if 'outc.0' in k:
                checkpoint['state_dict'][k.replace(
                    'outc.0', 'outc')] = checkpoint['state_dict'][k]
                del checkpoint['state_dict'][k]
        msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(msg)
        print("=> loaded checkpoint, starting at Epoch {}".format(start_epoch))

        result_dict, progress = err_validate_seq(val_loader, model, criterion,
                                                 cls_criterion, start_epoch,
                                                 args, writer)
        scalars = progress.tb_scalar_dict()
        torch.save((result_dict, scalars), res_fname.format(val_nsteps))


@torch.no_grad()
def err_validate_seq(val_loader,
                     model,
                     criterion,
                     cls_criterion,
                     epoch,
                     args,
                     writer=None):
    val_nsteps = val_loader.dataset._n_steps
    print('Validating {} steps'.format(val_nsteps))
    out_nhoods = [args.data.out_nhood]
    out_scales = [list(np.array(args.data.output_gridsize))]
    out_scales = ['X'.join([str(d) for d in out_scales[0]])]
    tbsuffixes = [
        '_evalsteps{}_{}_{}'.format(val_nsteps, out_nhoods[0], out_scales[0])
    ]
    batch_time = AverageMeter('Time', ':04.2f', tbnames=['val/time'])
    top1 = AverageMeter('Acc@1', '', tbnames=['val/acc' + tbsuffixes[0]])
    ious = AverageMeter('IOU', '', tbnames=['val/iou' + tbsuffixes[0]])
    cls_maps = AverageMeter('ClassMAP',
                            '',
                            tbnames=['val/cls_map' + tbsuffixes[0]])
    aps = AverageMeter('AP', '', tbnames=['val/ap' + tbsuffixes[0]])
    edge_aps = AverageMeter('E-AP', '', tbnames=['val/edgeap' + tbsuffixes[0]])
    progress = ProgressMeter(len(val_loader),
                             [batch_time, top1, ious, aps, edge_aps, cls_maps],
                             prefix='Test: ',
                             tbwriter=writer)
    epochmetrics = Evaluator()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (data, target, meta) in enumerate(val_loader):
        audio, rgb, relpath, target, meta = data_utils.prepare_batch(
            data, target, meta)
        # compute output
        finalpad = torch.FloatTensor([0])
        (output, cls_output,
         feat) = model(rgb=rgb,
                       audio=audio,
                       relpath=relpath,
                       padding=meta['sequence_padding'][0].item())
        print(rgb[0, 0, :, :])
        print(output[0, :, :, :].nonzero())

        if target.shape[2] < output.shape[2]:
            padding = (output.shape[2] - target.shape[2]) // 2
            target = F.pad(target, [padding] * 4)
            if 'predictable_target' in meta:
                meta['predictable_target'] = F.pad(meta['predictable_target'],
                                                   [padding] * 4)
            if 'semantic_target' in meta:
                meta['semantic_target'] = F.pad(meta['semantic_target'],
                                                [padding] * 4)
        elif output.shape[2] < target.shape[2]:
            padding = (target.shape[2] - output.shape[2]) // 2
            output = F.pad(output, [padding] * 4)
            cls_output = F.pad(cls_output, [padding] * 4)

        prob = F.softmax(output, 1)
        epochmetrics.append(output=output,
                            cls_output=cls_output,
                            target=target,
                            semantic_target=meta['semantic_target'],
                            predictable_target=meta['predictable_target'],
                            prob=prob)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            progress.display(i)
    epochmetrics.collect()
    (val_accs, val_aps, val_edge_aps,
     val_cls_map) = epochmetrics.get_all_metrics(
         balance=args.environment.evaluate_balanced)

    top1.update(np.nanmean(val_accs), 1)
    aps.update(np.nanmean(val_aps), 1)
    edge_aps.update(np.nanmean(val_edge_aps), 1)
    cls_maps.update(val_cls_map, 1)
    progress.display(len(val_loader))
    result_dict = {
        'accs': val_accs,
        'aps': val_aps,
        'edge_aps': val_edge_aps,
        'cls_map': val_cls_map
    }
    progress.tbwrite(int(epoch * 100000))
    scalars = progress.tb_scalar_dict()
    print(json.dumps(scalars, sort_keys=True, indent=2))
    return result_dict, progress


class Evaluator:
    def __init__(self):
        self.stored_data = defaultdict(list)

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if v.is_cuda:
                v = v.cpu()
            self.stored_data[k].append(v)

    def collect(self):
        for k in self.stored_data.keys():
            self.stored_data[k] = torch.cat(self.stored_data[k], 0)

    def get_all_metrics(self, balance=False):
        compute_edge_ap = models.utils.EdgeAP()
        output = self.stored_data['output']
        target = self.stored_data['target']
        prob = self.stored_data['prob']
        predictable_target = self.stored_data['predictable_target']
        pred = (output[:, 0] < output[:, 1]).long().detach()
        pred_scores = prob[:, 1].detach()
        pred_scores[predictable_target == 0] = pred_scores.min()

        print('Computing total metrics...')
        predictable_loc = (predictable_target > 0)
        clsacc1 = 0.0

        cls_output = self.stored_data['cls_output']
        cls_target = self.stored_data['semantic_target']
        cls_predictable_loc = (target > 0) * (cls_target > 0)

        num_elements = output.size(0)
        accs = []
        aps = []
        edge_aps = []
        clswise_aps = [0.0] * 13
        cls_prob = F.softmax(cls_output, 1)
        cls_pred = cls_prob.argmax(1)

        # Compute CLS AP
        interior_bool = (pred != 0)
        cls_prob = cls_prob * (
            interior_bool.unsqueeze(1).expand_as(cls_prob).float())
        clswise_aps = np.zeros((cls_prob.shape[0], 13))
        for i in range(len(cls_prob)):
            for c_i in range(1, 14):
                cls_pred_scores = cls_prob[i, c_i - 1]
                cls_pred_scores = cls_pred_scores.contiguous().view(-1)
                cls_i_gt = (cls_target[i] == c_i).view(-1).long()
                try:
                    cls_i_ap = data_utils.AP(
                        cls_pred_scores[cls_predictable_loc[i].view(-1)],
                        cls_i_gt[cls_predictable_loc[i].view(-1)],
                        balance=False)
                    clswise_aps[i, c_i - 1] = cls_i_ap
                except Exception as e:
                    clswise_aps[i, c_i - 1] = np.nan
        clswise_aps = np.nanmean(clswise_aps, axis=0)
        cls_map = np.nanmean(clswise_aps)

        print('Evaluating {} elements... '.format(num_elements))
        for i in range(num_elements):
            # Compute acc
            acc1 = data_utils.accuracy(
                output.permute(0, 2, 3, 1)[i].contiguous().view(
                    -1, 2)[predictable_loc[i].view(-1)],
                target[i].view(-1)[predictable_loc[i].view(-1)],
                topk=(1, ),
                balance=balance)[0][0].item()
            accs.append(acc1)

            # Compute AP
            ap = data_utils.AP(pred_scores[i][predictable_target[i] > 0],
                               target[i][predictable_target[i] > 0],
                               balance=balance)
            aps.append(ap)

            # Compute Edge AP
            edge_ap = compute_edge_ap(pred_scores[i:i + 1],
                                      target[i:i + 1],
                                      predictable_target[i:i + 1],
                                      balance=balance)
            edge_aps.append(edge_ap)

        return (accs, aps, edge_aps, cls_map)


@hydra.main(config_path='./configs/avmap/config.yaml')
def main(args):
    update_pythonpath_relative_hydra()
    ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                              'checkpoint_{:04d}.pth')
    if os.path.exists(ckpt_fname.format(args.optim.epochs - 1)) and (
            args.environment.evaluate_path != ''):
        print('{} training has finished'.format(args.logging.name))

    executor = submitit.AutoExecutor(
        folder=os.path.join(
            args.environment.project_dir, 'audioperception_data',
            'submitit_train_logs',
            '{}'.format(args.logging.name + args.logging.suffix)),
        max_num_timeout=100,
        cluster="debug",
    )
    executor.update_parameters(name=args.logging.name + args.logging.suffix)

    # Convert all paths to absolute
    args.logging.ckpt_dir = hydra_utils.to_absolute_path(args.logging.ckpt_dir)
    args.logging.tb_dir = hydra_utils.to_absolute_path(args.logging.tb_dir)
    args.data.train_env_list_file = hydra_utils.to_absolute_path(
        args.data.train_env_list_file)
    args.data.val_env_list_file = hydra_utils.to_absolute_path(
        args.data.val_env_list_file)
    args.data.full_eval_path = hydra_utils.to_absolute_path(
        args.data.full_eval_path)
    job = executor.submit(
        Worker(train_func=train_seq, val_func=err_validate_seq), args)
    job.result()


if __name__ == '__main__':
    main()
