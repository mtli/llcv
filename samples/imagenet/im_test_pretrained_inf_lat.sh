#!/bin/bash

# This scripts measures the inference latency
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"

scriptName=`basename "$0"`
expName=${scriptName%.*}


python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--out-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
		pretrained=pytorch \
	--test-init \
	--batch-size 1 \
	--n-worker 4 \
	--log-interval 500 \
	--inf-latency \
	--timing-iter 10000 \
