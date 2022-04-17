@echo off

:: This scripts measures the inference latency
:: Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

set "dataDir=D:\Data"
set "expName=%~n0"

:: We observe empirically that by limiting the threads,
:: timing becomes more stable, and the model runs faster
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set OMP_NUM_THREADS=1


python -m llcv.tools.test ^
	--exp-dir "%dataDir%\Exp\ImageNet\test\%expName%" ^
	--dataset ImageNet ^
	--data-root "%dataDir%\ILSVRC2012" ^
	--out-dir "%dataDir%\Exp\ImageNet\test\%expName%" ^
	--task ClsTask ^
	--model ResNet ^
	--model-opts ^
		depth=50 ^
		pretrained=pytorch ^
	--test-init ^
	--batch-size 1 ^
	--n-worker 4 ^
	--log-interval 500 ^
	--inf-latency ^
	--timing-iter 10000 ^
