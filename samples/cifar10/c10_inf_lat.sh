#!/bin/bash

# This scripts measures the inference latency
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

dataDir="/data3/mengtial"

# This script requires that you have a trained model using c10.sh
expName=c10


# We observe empirically that by limiting the threads,
# timing becomes more stable, and the model runs faster
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/CIFAR-10/train/${expName}" \
	--dataset CIFAR10 \
	--data-root "${dataDir}/SmallDB/CIFAR-10" \
	--out-dir "${dataDir}/Exp/CIFAR-10/test/${expName}" \
	--batch-size 1 \
	--n-worker 2 \
	--log-interval 500 \
	--inf-latency \
