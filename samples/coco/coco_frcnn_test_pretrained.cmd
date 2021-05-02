@echo off

set "dataDir=D:\Data"
set "expName=%~n0"


:: Currently, only single-GPU non-distributed testing is supported
python -m llcv.tools.test ^
	--exp-dir "%dataDir%\Exp\COCO\test\%expName%" ^
	--dataset COCODataset ^
	--data-root "%dataDir%\COCO\val2017" ^
	--data-opts ^
		ann_file="%dataDir%\COCO\annotations\instances_val2017.json" ^
	--task DetTask ^
	--out-dir "%dataDir%\Exp\COCO\test\%expName%" ^
	--model TVFasterRCNN ^
	--test-init ^
	--batch-size 1 ^
	--n-worker 2 ^
	--inf-latency ^
