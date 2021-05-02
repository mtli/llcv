@echo off

:: This scripts measures the inference latency
:: Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

:: This script requires that you have a trained model using c10.cmd
set "expName=c10"


python -m llcv.tools.test ^
	--exp-dir "%dataDir%\Exp\CIFAR-10\train\%expName%" ^
	--dataset CIFAR10 ^
	--data-root "%dataDir%\SmallDB\CIFAR-10" ^
	--out-dir "%dataDir%\Exp\CIFAR-10\test\%expName%" ^
	--batch-size 1 ^
	--n-worker 2 ^
	--log-interval 500 ^
	--inf-latency ^
