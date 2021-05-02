#!/bin/bash

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
	--batch-size 256 \
	--n-worker 8 \
