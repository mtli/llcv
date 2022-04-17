@echo off

:: This scripts measures the inference latency
:: Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

:: This script requires that you have a trained model using c10.cmd
set "dataDir=D:\Data"
set "expName=c10"

:: We observe empirically that by limiting the threads,
:: timing becomes more stable, and the model runs faster
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set OMP_NUM_THREADS=1


python -m llcv.tools.test ^
	--exp-dir "%dataDir%\Exp\CIFAR-10\train\%expName%" ^
	--dataset CIFAR10 ^
	--data-root "%dataDir%\SmallDB\CIFAR-10" ^
	--out-dir "%dataDir%\Exp\CIFAR-10\test\%expName%" ^
	--batch-size 1 ^
	--n-worker 2 ^
	--log-interval 500 ^
	--inf-latency ^
