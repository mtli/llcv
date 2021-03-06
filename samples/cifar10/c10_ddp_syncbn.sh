#!/bin/bash

dataDir="/data3/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}

nGPU=4

python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module \
 	llcv.tools.train \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--model-opts \
		norm_type=SyncBatchNorm \
	--n-epoch 200 \
	--batch-size 128 \
	--n-worker 4 \
	--gpu-gather \
	--train-gather \
	--log-env-info \
  && \
python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module \
 	llcv.tools.test \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--out-dir "${dataDir}/Exp/CIFAR-10/test/${expName}" \
	--model-opts \
		norm_type=SyncBatchNorm \
	--batch-size 4048 \
	--n-worker 8 \
	--gpu-gather \
