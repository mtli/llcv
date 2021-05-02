@echo off

set "dataDir=tmp"
set "expName=%~n0"

set "LLCV_EXT_PATH=%~dp0"
echo LLCV_EXT_PATH is set to "%LLCV_EXT_PATH%"

:: Here it uses current directory in this example
:: And you can be replace it with your own repo's directory


python -m llcv.tools.test ^
	--exp-dir "%dataDir%\%expName%" ^
	--dataset RandGen ^
	--data-root "%dataDir%" ^
	--out-dir "%dataDir%\%expName%" ^
	--task PrintTask ^
	--model Identity ^
	--test-init ^
	--batch-size 2 ^
	--n-worker 1 ^
