import logging

import torchvision.datasets as tv_datasets

from .coco import COCODataset
from ..pipelines import build_tv_pipelines
from ...utils import build_ext_class


def build_dataset(args, is_train):
    logging.info(f'Creating dataset {args.dataset}')
    
    dataset = build_ext_class('datasets', args.dataset, args)
    if dataset is not None:
        return dataset

    if args.dataset in vars(tv_datasets):
        ffcv_cvt = hasattr(args, 'ffcv_cvt') and args.ffcv_cvt
        if ffcv_cvt:
            pipelines = [None, None]

        kwargs = {}
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            kwargs['train'] = is_train
            kwargs['download'] = True
            if not ffcv_cvt:
                pipelines = build_tv_pipelines(args, is_train, 'CIFAR')
        elif args.dataset in ['ImageNet']:
            kwargs['split'] = 'train' if is_train else 'val'
            if not ffcv_cvt:
                pipelines = build_tv_pipelines(args, is_train, 'ImageNet')
        else:
            if not ffcv_cvt:
                pipelines = build_tv_pipelines(args, is_train)
        dataset = vars(tv_datasets)[args.dataset](
            root=args.data_root,
            transform=pipelines[0],
            target_transform=pipelines[1],
            **kwargs,
        )
    else:
        dataset = globals()[args.dataset](
           args, is_train=is_train)
    return dataset
