#!/bin/bash

dataDir="/data3/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}


python -m llcv.tools.train \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--n-epoch 200 \
	--batch-size 128 \
	--n-worker 4 \
	--log-env-info \
   && \
python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--out-dir "${dataDir}/Exp/CIFAR-10/test/${expName}" \
	--batch-size 2048 \
	--n-worker 8 \
