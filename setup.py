from setuptools import setup
from os.path import join, dirname, abspath

import re

re_ver = re.compile(r"__version__\s+=\s+'(.*)'")
with open(join(abspath(dirname(__file__)), 'llcv', '__init__.py'), encoding='utf-8') as f:
    version = re_ver.search(f.read()).group(1)

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

setup(
    name='llcv',
    version=version,
    description='A Modular and Extensible Framework for Computer Vision',
    long_description='See project page: https://github.com/mtli/llcv',
    url='https://github.com/mtli/llcv',
    author='Mengtian (Martin) Li',
    author_email='martinli.work@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='computer vision deep learning pytorch',
    packages=['llcv'],
    python_requires='>=3',
    install_requires=[
        'python-dateutil',
        'py-cpuinfo',
        'tqdm',
        'numpy',
        'pillow',
        'tensorboard',
        'linearlr',
    ],
    include_package_data = True,
)
