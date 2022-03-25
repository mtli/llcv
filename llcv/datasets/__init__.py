import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as tv_datasets

from .pipelines import build_pipeline
from ..utils import build_ext_class, dist_is_on
from .coco import COCODataset


def build_dataset(args, is_train):
    logging.info(f'Creating dataset {args.dataset}')
    
    dataset = build_ext_class('datasets', args.dataset, args)
    if dataset is not None:
        return dataset

    if args.dataset in vars(tv_datasets):
        kwargs = {}
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            kwargs['train'] = is_train
            kwargs['download'] = True
            pipeline = build_pipeline(args, is_train, 'CIFAR')
        elif args.dataset in ['ImageNet']:
            kwargs['split'] = 'train' if is_train else 'val'
            pipeline = build_pipeline(args, is_train, 'ImageNet')
        else:
            pipeline = build_pipeline(args, is_train)
        dataset = vars(tv_datasets)[args.dataset](
            root=args.data_root, transform=pipeline, **kwargs)
    else:
        dataset = globals()[args.dataset](
           args, is_train=is_train)
    return dataset

def build_loader(args, is_train):
    dataset = build_dataset(args, is_train)

    n_gpu = torch.cuda.device_count()
    if args.batch_size_per_gpu:
        args.batch_size = n_gpu*args.batch_size_per_gpu
    if args.n_worker_per_gpu:
        args.n_worker = n_gpu*args.n_worker_per_gpu

    if dist_is_on():
        if args.batch_size % n_gpu:
            raise ValueError(f'batch_size ({args.batch_size}) should be disivible by # of GPUs per distributed process ({n_gpu})')
        batch_size = args.batch_size//n_gpu
    
        if args.n_worker % n_gpu:
            raise ValueError(f'n_worker ({args.n_worker}) should be disivible by # of GPUs per distributed process ({n_gpu})')
        n_worker = args.n_worker//n_gpu
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        batch_size = args.batch_size
        n_worker = args.n_worker
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(is_train and sampler is None) or args.shuffle,
        num_workers=n_worker,
    )

    return loader
