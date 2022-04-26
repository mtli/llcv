#!/bin/bash

dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}


python -m llcv.tools.train \
	--exp-dir "${dataDir}/Exp/ImageNet/train/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
	--n-epoch 90 \
	--batch-size 256 \
	--n-worker 32 \
	--wd 1e-4 \
	--log-env-info \
	--reset \
	--fp16 \
  &&
python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/ImageNet/train/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--out-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
	--batch-size 256 \
	--n-worker 32 \
	--log-env-info \
