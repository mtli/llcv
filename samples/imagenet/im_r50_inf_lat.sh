#!/bin/bash

# This scripts measures the inference latency
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"

# This script requires that you have a trained model using im_r50.sh
expName=im_r50


# We observe empirically that by limiting the threads,
# timing becomes more stable, and the model runs faster
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/ImageNet/train/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--out-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
	--batch-size 1 \
	--n-worker 4 \
	--log-interval 500 \
	--inf-latency \
	--timing-iter 10000 \
