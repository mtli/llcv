#!/bin/bash

dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}

nGPU=8


python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module \
 	llcv.tools.train \
	--exp-dir "${dataDir}/Exp/ImageNet/train/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--ffcv \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
	--n-epoch 90 \
	--batch-size 256 \
	--n-worker 16 \
	--wd 1e-4 \
	--log-env-info \
  &&
python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module \
	llcv.tools.test \
	--exp-dir "${dataDir}/Exp/ImageNet/train/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--ffcv \
	--out-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
	--batch-size 512 \
	--n-worker 16 \
