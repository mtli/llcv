@echo off

set "dataDir=tmp"
set "expName=%~n0"


python -m llcv.tools.train ^
	--exp-dir "%dataDir%\Exp\%expName%" ^
	--data-root "%dataDir%\CIFAR-10" ^
  && ^
python -m llcv.tools.test ^
	--exp-dir "%dataDir%\Exp\%expName%" ^
	--data-root "%dataDir%\CIFAR-10" ^
	--out-dir "%dataDir%\Exp\%expName%_test" ^
