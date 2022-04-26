from .datasets import build_dataset
from .torch_loader import build_torch_loader

def build_dataset_and_loader(args, is_train):
    dataset = build_dataset(args, is_train)
    if hasattr(args, 'ffcv') and args.ffcv:
        from .ffcv_loader import build_ffcv_loader
        loader = build_ffcv_loader(args, is_train)
    else:
        loader = build_torch_loader(args, dataset, is_train)
    return dataset, loader
