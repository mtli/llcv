import numpy as np

import torch

from ffcv.fields.decoders import \
    SimpleRGBImageDecoder, IntDecoder, \
    RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, RandomTranslate, ToTorchImage

def build_ffcv_pipelines(args, is_train, dataset_group=None):
    if dataset_group == 'CIFAR':
        mean = 255*np.array([0.4914, 0.4822, 0.4465])
        std = 255*np.array([0.2023, 0.1994, 0.2010])
        image_pipeline = [SimpleRGBImageDecoder()]
        if is_train:
            image_pipeline += [
                RandomTranslate(padding=2, fill=tuple(mean.astype(np.int).tolist())),
                RandomHorizontalFlip(),
            ] 
        image_pipeline.append(ToTensor())
        if args.cuda:
            image_pipeline.append(ToDevice(torch.cuda.current_device(), non_blocking=True))
        image_pipeline += [
            ToTorchImage(),
            NormalizeImage(mean, std, np.float32),
        ]
    elif dataset_group == 'ImageNet':
        mean = 255*np.array([0.485, 0.456, 0.406])
        std = 255*np.array([0.229, 0.224, 0.225])
        if is_train:
            decoder = RandomResizedCropRGBImageDecoder((224, 224))
            image_pipeline = [decoder, RandomHorizontalFlip()]
        else:
            decoder = CenterCropRGBImageDecoder((224, 224), ratio=224/256)
            image_pipeline = [decoder]

        image_pipeline.append(ToTensor())
        if args.cuda:
            image_pipeline.append(ToDevice(torch.cuda.current_device(), non_blocking=True))
        image_pipeline += [
            ToTorchImage(),
            NormalizeImage(mean, std, np.float32),
        ]
    else:
        pipelines = [None, None]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]
    if args.cuda:
        label_pipeline.append(ToDevice(torch.cuda.current_device(), non_blocking=True))
    pipelines = {'image': image_pipeline, 'label': label_pipeline}
    return pipelines
