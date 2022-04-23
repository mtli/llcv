import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import build_dataset
from ..utils import dist_is_on


def build_torch_loader(args, is_train):
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
