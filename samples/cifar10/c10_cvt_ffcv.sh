#!/bin/bash

dataDir="/data3/mengtial"

python -m llcv.tools.ffcv_convert_data \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
