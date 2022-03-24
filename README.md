# LLCV &mdash; An Extensible Framework for **L**ow-**L**atency **C**omputer **V**ision Research


![#fc4903](https://via.placeholder.com/15/fc4903/000000?text=+) Research-oriented &mdash; flexible interfaces and dynamic modules

![#3865cf](https://via.placeholder.com/15/3865cf/000000?text=+) Latency-oriented &mdash; efficient implementation and various types of timing support

Features:
- Training auto-resume after interruption
- Support for DataParellel & DistributedDataParallel
- Tensorboard integration

Supported tasks:
- Image classification
- Object detection

Notable models:
- ResNet-18 on CIFAR-10 (Top-1 95.62%)
- ResNet-50 on ImageNet (Top-1 76.56%, Top-5 93.02%)
- Faster R-CNN on COCO (AP 37.0, through torchvision)

Note that this repo is currently under active development and compatibility is not guaranteed. While there are many standard techniques for faster implementations, this repo mostly focuses on the research aspect of reducing training and inference cost.


## Installation

[![PyPI version](https://badge.fury.io/py/llcv.svg)](https://badge.fury.io/py/llcv)

Install [PyTorch](https://pytorch.org/) (>= 1.1) and run:
```
pip install llcv
```

If you prefer to install it in develop mode:
```
git clone https://github.com/mtli/llcv
cd llcv
pip install -e .
```


## Getting Started

Check out the scripts in the `samples/` for various tasks.

To use this framework for your own projects, you can either fork this repo, or using this repo as a dependency through [dynamic modules](doc/design.md#dynamic-modules).

You can find more details about the framework's design [here](doc/design.md).


## Model Zoo

Note that latency and throughput are measured on Geforce GTX 1080 Ti GPUs, unless otherwise stated.

<br>

### CIFAR-10 (Classification)
| ID  |    Method    | Epc | Top-1 (%)  | E2E (ms) | Inf (ms) | Train BS | Train TP | Train Time | Ckpt | Log | Script |
| :-: | :----------: | :-: | :----: | :------: | :------: | :------: | :------: | :--------: | :--: | :-: | :----: |
| 1   | ResNet-18    | 200 | 95.62 |        - | 3.05 ± 0.302 |  128 |   2081Hz |      1h22m | [ckpt](https://www.cs.cmu.edu/~mengtial/proj/llcv/model_zoo/c10-r18-e200-95.62-870a16f.pth) | [log](https://www.cs.cmu.edu/~mengtial/proj/llcv/model_zoo/c10-r18-e200-95.62-870a16f.log) | [script](samples/cifar10/c10.sh) |

In the above table, timings are measured on a single GPU for both training and testing.

<br>

### ImageNet (Classification)
| ID  |    Method    | Epc | Top-1 (%) | Top-5 (%) | E2E (ms) |  Inf (ms) | Train BS | Train TP | Train Time | Ckpt | Log | Script |
| :-: | :----------: | :-: | :-------: | :-------: |:-------: | :-------: | :------: | :------: | :--------: | :--: | :-: | :----: |
| 1   | ResNet-50    | 90  |     76.56 |     93.02 |        - | 9.82 ± 3.59 |    256 |  504.8Hz |      2d16h | [ckpt](https://www.cs.cmu.edu/~mengtial/proj/llcv/model_zoo/im-r50-e90-76.56-870a16f.pth) | [log](https://www.cs.cmu.edu/~mengtial/proj/llcv/model_zoo/im-r50-e90-76.56-870a16f.log) | [script](samples/imagenet/im_r50.sh) |

In the above table, timings are measured on 4 GPUs for training and a single GPU for testing.

<br>

### COCO (Detection)

| ID  |    Method    | Epc |  AP  | E2E (ms) |  Inf (ms) | Train BS | Train TP | Train Time | Ckpt | Log | Script |
| :-: | :----------: | :-: | :--: |:-------: | :-------: | :------: | :------: | :--------: | :--: | :-: | :----: |
| 1   | Faster R-CNN R50 FPN (torchvision)  | - | 37.0 | - | 75.2 ± 54.4 | - | - |        - | -    | -   | [script](samples/coco/coco_frcnn_test_pretrained.sh) |

In the above table, timings are measured on 8 GPUs for training and a single GPU for testing.
