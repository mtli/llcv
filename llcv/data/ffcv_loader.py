from os.path import join

import torch

from ffcv.loader import Loader, OrderOption

from .pipelines.ffcv_pipelines import build_ffcv_pipelines
from ..utils import dist_is_on


def build_ffcv_loader(args, is_train):
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        dataset_group = 'CIFAR'
    elif args.dataset in ['ImageNet']:
        dataset_group = 'ImageNet'
    pipelines = build_ffcv_pipelines(args, is_train, dataset_group)

    n_gpu = torch.cuda.device_count()
    if args.batch_size_per_gpu:
        args.batch_size = n_gpu*args.batch_size_per_gpu
    if args.n_worker_per_gpu:
        args.n_worker = n_gpu*args.n_worker_per_gpu

    distributed = dist_is_on()
    if distributed:
        if args.batch_size % n_gpu:
            raise ValueError(f'batch_size ({args.batch_size}) should be disivible by # of GPUs per distributed process ({n_gpu})')
        batch_size = args.batch_size//n_gpu
    
        if args.n_worker % n_gpu:
            raise ValueError(f'n_worker ({args.n_worker}) should be disivible by # of GPUs per distributed process ({n_gpu})')
        n_worker = args.n_worker//n_gpu
    else:
        batch_size = args.batch_size
        n_worker = args.n_worker

    loader = Loader(
        join(args.data_root, ('train' if is_train else 'val') + '.beton'),
        batch_size=batch_size, 
        num_workers=n_worker,
        order=OrderOption.RANDOM if is_train or args.shuffle else OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=is_train,
        pipelines=pipelines,
        distributed=distributed,
    )

    return loader
