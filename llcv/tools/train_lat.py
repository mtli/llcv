'''
A script for measuring the latency of training iterations, or more precisely,
the latency of a forward and a backward pass during training. While train.py
provides timing utility as well, it is used to measure the training
throughput. 

This script runs on a single GPU without dataloader prefetching.
'''

import logging

import numpy as np
import torch

from ..datasets import build_loader
from ..tasks import build_task
from ..utils import get_default_parser, env_setup, \
    Timer, get_eta, get_batchsize


def add_args(parser):
    ## Basic options
    parser.add_argument('--dataset', type=str, default='CIFAR10',
        help='dataset')
    parser.add_argument('--data-root', type=str, required=True,
        help='root directory of the dataset')
    parser.add_argument('--n-epoch', type=int, default=20,
        help='# of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size for training (per node)')
    parser.add_argument('--n-worker', type=int, default=0,
        help='# of workers for data prefetching (per node)')
    parser.add_argument('--lr', type=float, default=0.1,
        help='base learning rate (default: 0.1)')

    ## Hyperparameters
    parser.add_argument('--optim', type=str, default='SGD',
        help='optimizer (default: SGD)')
    parser.add_argument('--wd', type=float, default=5e-4,
        help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='optimizer momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', default=False,
        help='enables nesterov momentum')
    parser.add_argument('--lr-schedule', type=str, default='Linear',
        help='learning rate schedule (default: Linear)')
    parser.add_argument('--lr-update-per-epoch', action='store_true', default=False,
        help='update learning rate after each epoch instead of each iter by default')
    parser.add_argument('--lr-decay-epoch', type=int, default=50,
        help='learning rate schedule (default: 50)')
    parser.add_argument('--lr-schedule-gamma', type=float, default=0.1,
        help='intepretation depends on lr_schedule (default: 0.1)')

    ## Training Settings
    parser.add_argument('--reset', action='store_true', default=False,
        help='DANGER: purge the exp_dir and start a fresh new training run')
    parser.add_argument('--pretrain', type=str, default=None,
        help='pretrained weights')
    parser.add_argument('--resume-epoch', type=int, default=None,
        help='by default, the resume epoch is automatically determined from '
             'the checkpoint file and this option overwrites the default')
    parser.add_argument('--no-resume-load', action='store_false', dest='resume_load', default=True,
        help='resume from an epoch, but not load the checkpoint')
    parser.add_argument('--batch-size-per-gpu', type=int, default=None,
        help='alternative to batch_size (and overrides it)')
    parser.add_argument('--n-worker-per-gpu', type=int, default=None,
        help='alternative n_worker (and overrides it)')
    parser.add_argument('--epoch-iter', type=int, default=float('inf'),
        help='maximum # of iterations per epoch')
    parser.add_argument('--log-interval', type=int, default=50,
        help='after every how many iters to log the training status')
    parser.add_argument('--save-interval', type=int, default=5,
        help='after every how many epochs to save the learned model')

def main():
    ## Overall timer
    tmr_main = Timer()

    ## Argument parser and environment setup
    parser = get_default_parser('llcv - training latency script')
    add_args(parser)
    args = env_setup(parser, 'train-lat', ['data_root', 'pretrain'])
    
    ## Prepare the dataloader
    train_loader = build_loader(args, is_train=True)
    logging.info(f'# of classes: {len(train_loader.dataset.classes)}')
    n_train = len(train_loader.dataset)
    logging.info(f'# of training examples: {n_train}')
    assert n_train
    if args.epoch_iter < len(train_loader):
        logging.warning(
            f'The number of iterations per epoch is limited to {args.epoch_iter}')
        train_epoch_iter = args.epoch_iter
    else:
        train_epoch_iter = len(train_loader)
    if args.to_cuda_before_task:
        device = torch.cuda.current_device()

    ## Initialize task
    task = build_task(args, train_loader, is_train=True)

    if task.resume_epoch >= args.n_epoch:
        logging.warning(f'The model is already trained for {task.resume_epoch} epochs')
        return

    ## Start training
    last_saved_epoch = task.resume_epoch # resume_epoch is by default 0
    # counters for ETA
    n_iter_epoch = 0
    n_iter_total = (args.n_epoch - task.resume_epoch)*train_epoch_iter
    # latency measures 
    t_valid_total = 0
    t_valid_cnt = 0
    timing_samples = []
    is_warmup_sample = []

    task.train_mode(gather=False)
    logging.info('Training starts')
    tmr_train = Timer()
    for epoch in range(task.resume_epoch + 1, args.n_epoch + 1):
        n_seen = 0
        n_warpup = 0
        t_warmup = 0
        
        tmr_epoch = Timer()
        for i, data in enumerate(train_loader):
            i += 1
            # the last batch can be smaller than normal
            this_batch_size = get_batchsize(data)

            if args.to_cuda_before_task:
                if isinstance(data, (list, tuple)):
                    data = [x.to(device) for x in data]
                else:
                    data = data.to(device)

            torch.cuda.synchronize()
            tmr_iter = Timer()

            task.forward(data)
            task.backward()
            tmr_iter.stop()

            if not args.lr_update_per_epoch:
                task.update_lr_iter()

            n_seen += this_batch_size

            torch.cuda.synchronize()
            t_iter = tmr_iter.elapsed()
            timing_samples.append(t_iter)

            if i <= args.timing_warmup_iter:
                is_warmup_sample.append(True)
                n_warpup += this_batch_size
                t_warmup += t_iter
            else:
                is_warmup_sample.append(False)
                t_valid_total += t_iter
                t_valid_cnt += 1

                if i % args.log_interval == 0:
                    task.log_iter(
                        'train e%d: %4d/%4d, latency: %5.4gms, ' % 
                        (epoch, i, train_epoch_iter, 1e3*(t_valid_total/t_valid_cnt)),
                        ', ETA: ' + get_eta(tmr_train.check(), n_iter_epoch + i, n_iter_total),
                    )
                task.log_iter_tb(
                    (epoch-1)*train_epoch_iter + i,
                    is_train=True,
                )

            if i >= train_epoch_iter:
                break

        task.log_epoch(f'train e{epoch} summary: ')
        task.log_epoch_tb(epoch, is_train=True)
        task.reset_epoch()      

        tmr_epoch.stop()
        logging.info('end of epoch %d/%d: epoch time: %s, ETA: %s' %
                (epoch, args.n_epoch, tmr_epoch.elapsed(to_str=True),
                get_eta(tmr_train.check(),
                    epoch - task.resume_epoch,
                    args.n_epoch - task.resume_epoch,
                ))
        )
        
        if last_saved_epoch != epoch and epoch % args.save_interval == 0:
            task.save(epoch)
            last_saved_epoch = epoch
            
        if args.lr_update_per_epoch:
            task.update_lr_epoch()

        n_iter_epoch += train_epoch_iter

    if last_saved_epoch != args.n_epoch:
        # saving the last epoch if n_epoch is not divisible by save_interval
        task.save(args.n_epoch)
    
    timing_samples = np.asarray(timing_samples)
    is_warmup_sample = np.asarray(is_warmup_sample)
    task.summarize_timing(
        'training latency',
        timing_samples[np.logical_not(is_warmup_sample)],
        n_warmup=0,
        out_dir=args.exp_dir,
    )

    tmr_main.stop()
    logging.info(f'Training finished with total elapsed time {tmr_main.elapsed(to_str=True)}')


if __name__ == '__main__':
    main()
