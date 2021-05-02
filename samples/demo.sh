#!/bin/bash

dataDir="tmp"

scriptName=`basename "$0"`
expName=${scriptName%.*}


python -m llcv.tools.train \
	--exp-dir "${dataDir}/Exp/${expName}" \
	--data-root "${dataDir}/CIFAR-10" \
   && \
python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/${expName}" \
	--data-root "${dataDir}/CIFAR-10" \
	--out-dir "${dataDir}/Exp/${expName}_test" \
