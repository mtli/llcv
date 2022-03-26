from setuptools import setup

try:
    import torch, torchvision
except ModuleNotFoundError:
    raise Exception(
        'llcv requires PyTorch and torchvision,'
        'please follow the instructions on https://pytorch.org/'
        'to install them'
    )

torch_ver = [int(x) for x in torch.__version__.split('.')[:2]]
assert torch_ver >= [1, 1], 'llcv requires PyTorch >= 1.1'


setup()
