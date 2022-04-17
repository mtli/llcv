import logging

import numpy as np
import torch

from ..datasets import build_loader
from ..tasks import build_task
from ..utils import get_default_parser, env_setup, \
    Timer, get_eta, dist_get_world_size, get_batchsize


def add_args(parser):
    ## Basic options
    parser.add_argument('--dataset', type=str, default='CIFAR10',
        help='dataset')
    parser.add_argument('--data-root', type=str, required=True,
        help='root directory of the dataset')
    parser.add_argument('--batch-size', type=int, default=256,
        help='batch size for testing')
    parser.add_argument('--n-worker', type=int, default=8,
        help='# of workers for data prefetching (per node)')
    parser.add_argument('--ckpt-epoch', type=int,
        help='load the checkpoint of the specified epoch instead of the best one by default')
    parser.add_argument('--ckpt', type=str,
        help='path for the pre-trained model')
    parser.add_argument('--out-dir', type=str,
        help='output directory for results and summary')
    parser.add_argument('--test-init', action='store_true',
        help='test using initial weights without loading a checkpoint')
    parser.add_argument('--batch-size-per-gpu', type=int, default=None,
        help='alternative to batch_size (and overrides it)')
    parser.add_argument('--n-worker-per-gpu', type=int, default=None,
        help='alternative n_worker (and overrides it)')
    parser.add_argument('--log-interval', type=int, default=20,
        help='after every how many iters to log the testing status')
    parser.add_argument('--inf-latency', action='store_true',
        help='measures the inference latency in synchronized mode')
    parser.add_argument('--timing-iter', type=int, default=float('inf'),
        help='enters timing-only mode and only tests over a specified number of iterations '
             'instead of the whole dataset (this takes effect only when --inf-latency is on).'
    )

def main():
    ## Overall timer
    tmr_main = Timer()

    ## Argument parser and environment setup
    parser = get_default_parser('llcv - testing script')
    add_args(parser)
    args = env_setup(parser, 'test', ['data_root', 'ckpt', 'out_dir'])

    ## Prepare the dataloader
    test_loader = build_loader(args, is_train=False)
    n_test = len(test_loader.dataset)
    logging.info(f'# of testing examples: {n_test}')
    if args.to_cuda_before_task:
        device = torch.cuda.current_device()

    ## Initialize task
    task = build_task(args, test_loader, is_train=False)
    task.test_mode(gather=not args.inf_latency)

    ## Start testing
    n_seen = 0
    n_warpup = 0
    n_test_itr = len(test_loader)
    if args.inf_latency:
        logging.info(
            'Timing mode is enabled. Please ensure that no other resource-intensive processes '
            'are running on the same machine.'
        )
        if args.timing_iter < n_test_itr:
            n_test_itr = args.timing_iter
            logging.info(
                f'Timing-only mode enabled, and testing will '
                f'exit after {n_test_itr} iterations.'
            )
        timing_samples = np.empty(n_test_itr)

    speed_ratio = dist_get_world_size()
    logging.info('Testing starts')
    tmr_test = Timer()
    for i, data in enumerate(test_loader):
        i += 1
        if i > n_test_itr:
            break

        # the last batch can be smaller than normal
        this_batch_size = get_batchsize(data)

        if args.to_cuda_before_task:
            if isinstance(data, (list, tuple)):
                data = [x.to(device) for x in data]
            else:
                data = data.to(device)

        if args.inf_latency:
            torch.cuda.synchronize()
            tmr_iter = Timer()

        with torch.no_grad():
            task.forward(data)

        if args.inf_latency:
            torch.cuda.synchronize()
            tmr_iter.stop()
            timing_samples[i - 1] = tmr_iter.elapsed()

        n_seen += this_batch_size

        if i <= args.timing_warmup_iter:
            n_warpup += this_batch_size
        if i == args.timing_warmup_iter:
            t_warmup = tmr_test.check()

        if i % args.log_interval == 0:
            t_total = tmr_test.check()
            if i <= args.timing_warmup_iter:
                ave_speed = n_seen/t_total if t_total else float('inf')
            else:
                ave_speed = (n_seen - n_warpup)/(t_total - t_warmup)if (t_total - t_warmup) else float('inf')
            ave_speed *= speed_ratio

            prefix = 'test: %4d/%4d, %5.4gHz, ' % (i, n_test_itr, ave_speed)
            if args.inf_latency:
                start = 0 if i <= args.timing_warmup_iter else args.timing_warmup_iter
                prefix += '%5.4gms, ' % \
                    (1e3*timing_samples[start:i].mean())
            task.log_iter(prefix,  ', ETA: ' + get_eta(t_total, i, n_test_itr))

    if args.inf_latency:
        task.summarize_timing('inference latency', timing_samples, args.timing_warmup_iter, args.out_dir)
    else:
        task.dist_gather(is_train=False)
        task.summarize_test(args)

    tmr_main.stop()
    logging.info(f'Testing finished with total elapsed time {tmr_main.elapsed(to_str=True)}')

if __name__ == '__main__':
    main()
