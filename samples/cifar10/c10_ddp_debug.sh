#!/bin/bash

dataDir="/data3/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}

nGPU=3
# using an odd number here for easy discovery of bugs
# please set CUDA_VISIBLE_DEVICES correspondingly

python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module \
 	llcv.tools.train \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--n-epoch 20 \
	--batch-size 333 \
	--n-worker 9 \
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
	--batch-size 666 \
	--gpu-gather \
	--n-worker 9 \
  && \
python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--out-dir "${dataDir}/Exp/CIFAR-10/test/${expName}" \
	--batch-size 666 \
	--n-worker 9 \

# distributed training, distributed testing, and then a standard testing,
# which is used to validate the distributed results
