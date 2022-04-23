from .tv_pipelines import build_tv_pipelines

def build_pipelines(args, is_train, dataset_group=None):
    if hasattr(args, 'ffcv_cvt') and args.ffcv_cvt:
        pipelines = [None, None]
    elif hasattr(args, 'ffcv') and args.ffcv:
        from .ffcv_pipelines import build_ffcv_pipelines
        pipelines = build_ffcv_pipelines(args, is_train, dataset_group)
    else:
        pipelines = build_tv_pipelines(args, is_train, dataset_group)
    return pipelines
