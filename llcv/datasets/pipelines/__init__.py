import torchvision.transforms as tv_transforms


def build_pipeline(args, is_train, special=None):
    if special == 'CIFAR':
        normalize = tv_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        if is_train:
            pipeline = tv_transforms.Compose([
                tv_transforms.RandomCrop(32, padding=4),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                normalize,
            ])
        else:
            pipeline = tv_transforms.Compose([
                tv_transforms.ToTensor(),
                normalize,
            ])
    elif special == 'ImageNet':
        normalize = tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if is_train:
            pipeline = tv_transforms.Compose([
                tv_transforms.RandomResizedCrop(224),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                normalize,
            ])
        else:
            pipeline = tv_transforms.Compose([
                tv_transforms.Resize(256),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                normalize,
            ])
    else:
        pipeline = None

    return pipeline