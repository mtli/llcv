import argparse, logging
from os.path import join

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

from ..data import build_dataset
from ..utils import args_expanduser, OptsAction, Timer


def get_parser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    parser.add_argument('--dataset', type=str, default='ImageNet',
        help='dataset')
    parser.add_argument('--data-root', type=str, required=True,
        help='root directory of the dataset')
    parser.add_argument('--data-opts', nargs='+', action=OptsAction, default={},
        help='data options in a dict')

    return parser

def main():
    ## Overall timer
    tmr_main = Timer()

    ## Argument parser and environment setup
    parser = get_parser('llcv - dataset conversion')
    args = parser.parse_args()
    args_expanduser(args, ['data_root'])

    args.ffcv_cvt = True
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        writer_opts = {
            'image': RGBImageField(),
            'label': IntField(),
        }
    elif args.dataset in ['ImageNet']:
        writer_opts = {
            'image': RGBImageField(max_resolution=256, jpeg_quality=90),
            'label': IntField(),
        }
    else:
        raise ValueError('Unsupported dataset type for conversion: ' + args.dataset)

    for is_train in [True, False]:
        ## Build the dataset
        dataset = build_dataset(args, is_train=is_train)

        out_path = join(args.data_root, ('train' if is_train else 'val') + '.beton')
        ## Convert the dataset
        DatasetWriter(out_path, writer_opts).from_indexed_dataset(dataset)

    tmr_main.stop()
    logging.info(f'Conversion finished with total elapsed time {tmr_main.elapsed(to_str=True)}')

if __name__ == '__main__':
    main()
