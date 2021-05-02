import logging

from .backbones import *
from .detectors import *

from ..utils import build_ext_class


def build_model(args, dataset):
    logging.info('Creating model ' + args.model)

    model = build_ext_class('models', args.model, args, dataset)
    if model is not None:
        return model
        
    return globals()[args.model](args, dataset)
