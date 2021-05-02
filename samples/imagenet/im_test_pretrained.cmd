@echo off

set "dataDir=D:\Data"
set "expName=%~n0"


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
	--batch-size 256 ^
	--n-worker 8 ^
