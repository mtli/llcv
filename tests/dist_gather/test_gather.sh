#!/bin/bash

dataDir=/tmp

scriptName=`basename "$0"`
expName=${scriptName%.*}

export LLCV_EXT_PATH=`dirname "$0"`
echo LLCV_EXT_PATH is set to "${LLCV_EXT_PATH}"


nGPU=4

python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	--module llcv.tools.test \
	--exp-dir "${dataDir}/${expName}" \
	--dataset RangeDataset \
	--data-root "${dataDir}" \
	--out-dir "${dataDir}/${expName}" \
	--task PrintTask \
	--model Identity \
	--test-init \
	--batch-size-per-gpu 2 \
	--n-worker-per-gpu 1 \
	# --gpu-gather \
