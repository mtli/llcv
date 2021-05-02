import logging

from .cls import ClsTask
from .det import DetTask

from ..utils import build_ext_class


def build_task(args, loader, is_train):
    logging.info(f'Creating task {args.task}')

    task = build_ext_class('tasks', args.task, args, loader, is_train)
    if task is not None:
        return task
    return globals()[args.task](args, loader, is_train)
