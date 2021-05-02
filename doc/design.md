# Design

## Modules

- `tools`: top-level scripts for running different jobs (see the section below for details)
- `tasks`: various computer vision tasks
- `datasets`: dataset, loaders and augmentations
- `models`: neural networks
- `utils`: shared utility functions

## Tools
- `tools/train.py`: standard model training
- `tools/test.py`: standard model testing

## Dynamic Modules

The dynamic module design allows easy integration with external components from other projects. You can simply set the environment variable `LLCV_EXT_PATH` to the folder containing your custom modules, and llcv will load them automatically. An example is provided in `tests/custom_ext/`. Note that you can add multiple paths in `LLCV_EXT_PATH` just as you would for the `PATH` variable.
