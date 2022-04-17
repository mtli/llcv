import sys, argparse, logging, json, random
from os import name as os_name
from os import makedirs, environ
from os.path import join, expanduser, isdir, dirname, basename, isfile
from shutil import rmtree
from socket import gethostname

import numpy as np
import torch

from .. import __version__
from .stats import get_timestamp
from .dist import dist_get_rank, dist_add_args, dist_init, dist_get_info


def get_env_info(use_cuda=True, to_str=False):
    env_info = {
        'hostname': gethostname(),
    }

    ## Hardware info
    from cpuinfo import get_cpu_info
    env_info['CPU(s)'] = get_cpu_info()['brand_raw']

    env_info['use CUDA'] = use_cuda
    if use_cuda:
        env_info['CUDA_VISIBLE_DEVICES'] = environ.get('CUDA_VISIBLE_DEVICES', '(unset)')
        gpu_names = []
        all_same_gpus = True
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            if i > 0 and gpu_name not in gpu_names:
                all_same_gpus = False
            gpu_names.append(gpu_name)
        if all_same_gpus:
            n_gpu = len(gpu_names)
            env_info['GPU(s)'] = gpu_names[0] if n_gpu == 1 else f'({n_gpu}x) ' + gpu_names[0]
        else:
            env_info['GPU(s)'] = '(heterogeneous)\n' + \
                '\n'.join([f'  {i}. {gpu_name}' for i, gpu_name in enumerate(gpu_names)])

    ## Software info
    env_info['llcv'] = __version__
    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch Build'] = torch.__config__.show()

    try:
        import torchvision
        env_info['torchvision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info.update(dist_get_info())

    if to_str:
        return '- ' + '\n- '.join([(f'{k}: {v}') for k, v in env_info.items()]) + '\n'
    else:
        return env_info

def args_expanduser(args, path_fields):
    for p in path_fields:
        if vars(args)[p]:
            if isinstance(vars(args)[p], list):
                vars(args)[p] = [expanduser(_) for _ in vars(args)[p]]
            vars(args)[p] = expanduser(vars(args)[p])

class OptsAction(argparse.Action):
    """
    A custom argparse action to parse advanced options, and it
    supports multiple command line forms:
    1. --<option_name> <path_to_config_file>
        Supported formats are .py, and .json.
    2. --<option_name> <key1>=<val1> <key2>=<val2> ...
        <key> -> <k1>
              -> <k1>.<k2>.<k3>, ...
        <val> -> <v1>
              -> <v1>,<v2>,<v3>, ...
        <v1>  -> <True|False>
              -> <int>
              -> <float>
              -> <str>
    In all forms, a dict containing parsed options will be returned
    """

    @staticmethod
    def _parse_singleton(val):
        if val == 'True':
            return True
        elif val == 'False':
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        # val is a str at this point
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1 and '=' not in values[0]:
            cfg_path = values[0]
            if not isfile(cfg_path):
                raise FileNotFoundError(f'Config file "{cfg_path}" not found')
            if cfg_path.endswith('.py'):
                cfg_dir = dirname(cfg_path)
                cfg_name = basename(cfg_path)[:-3]
                sys.path.insert(0, cfg_dir)
                cfg_module = __import__(cfg_name)
                sys.path.pop(0)
                opts = {
                    k: v for k, v in vars(cfg_module).items() if not k.startswith('__')
                }
                del sys.modules[cfg_name]
            elif cfg_path.endswith('.json'):
                opts = json.load(open(cfg_path))
        else:
            opts = {}
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                val = [self._parse_singleton(v) for v in val.split(',')]
                if len(val) == 1:
                    val = val[0]
                if '.' in key:
                    ks = key.split('.')
                    if ks[0] in opts:
                        p = opts[ks[0]]
                        for i in range(1, len(ks)):
                            if ks[i] in p:
                                p = p[ks[i]]
                            else:
                                break
                    else:
                        p = opts
                        i = 0
                    d = {ks[-1]: val}
                    for k in reversed(ks[i:-1]):
                        d = {k: d}
                    p.update(d)
                else:
                    opts[key] = val
        setattr(namespace, self.dest, opts)

def get_default_parser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    parser.add_argument('--exp-dir', type=str, required=True, help='experiment directory')
    # parser.add_argument('--exp-dir', type=str, default='debug', help='experiment directory')
    parser.add_argument('--task', type=str, default='ClsTask',
        help='task')
    parser.add_argument('--data-opts', nargs='+', action=OptsAction, default={},
        help='data options in a dict')
    parser.add_argument('--model', type=str, default='ResNet',
        help='network model')
    parser.add_argument('--model-opts', nargs='+', action=OptsAction, default={},
        help='model options in a dict')
    parser.add_argument('--load-strict', action='store_true',
        help='enable strict matching for loading checkpoints')
    parser.add_argument('--log-level', type=str, default='INFO',
        help='log level (default: INFO)')
    parser.add_argument('--seed', type=int, default=None,
        help='random seed (default: None)')
    parser.add_argument('--shuffle', action='store_true', default=False,
        help='force shuffle even during testing (useful for timing)')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True,
        help='disable CUDA')
    parser.add_argument('--to-cuda-before-task', action='store_true', default=False,
        help='move to GPU in the top-level script (for timing purposes)')
    parser.add_argument('--gpu-gather', action='store_true',
        help='use GPU to gather outputs over the entire dataset (keep it off for large datasets)')
    parser.add_argument('--no-cudnn-benchmark', action='store_false', dest='cudnn_benchmark', default=True,
        help='disable cuDNN benchmark for size-varying tasks (e.g., variable image resolution and two-stage detectors)')
    parser.add_argument('--log-env-info', action='store_true', default=False,
        help='log the environment info at the beginning of the program')
    parser.add_argument('--no-ext', action='store_false', dest='ext', default=True,
        help='disable custom extensions')
    parser.add_argument('--timing-warmup-iter', type=int, default=5,
        help='# of iterations to be ignored at the beginning for '
             'timing purposes',
    )

    dist_add_args(parser)

    '''
    Note that the selection of GPU(s) should be done through the
    environment variable CUDA_VISIBLE_DEVICES. You might find the
    aliases below useful (append them to ~/.bash_aliases):

    alias showgpu='printenv CUDA_VISIBLE_DEVICES'
    alias unsetgpu='unset CUDA_VISIBLE_DEVICES'
    function gpu()
    {
        export CUDA_VISIBLE_DEVICES=$1
    }

    '''

    return parser

class LevelSpecificFormatter(logging.Formatter):
    def __init__(self, rank):
        if rank <= 0:
            # non-distributed or master
            self.def_fmt = '%(asctime)s - %(message)s'
            self.war_fmt = '%(asctime)s - WARNING - %(message)s'
            self.err_fmt = '%(asctime)s - *ERROR* - %(message)s'
        else:
            self.def_fmt = f'%(asctime)s [rank {rank}] - %(message)s'
            self.war_fmt = f'%(asctime)s [rank {rank}] - WARNING - %(message)s'
            self.err_fmt = f'%(asctime)s [rank {rank}] - *ERROR* - %(message)s'
        super().__init__(
            fmt=self.def_fmt,
            datefmt='%Y-%m-%d %H:%M:%S',
        )  
    
    def format(self, record):
        _fmt = self._style._fmt

        if record.levelno == logging.WARNING:
            self._style._fmt = self.war_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = self.err_fmt
        result = logging.Formatter.format(self, record)

        self._style._fmt = _fmt
        return result
        
def env_setup(parser, task_name='', path_fields=[]):
    args = parser.parse_args()
    if 'exp_dir' not in path_fields:
        path_fields.append('exp_dir')
    args_expanduser(args, path_fields)

    if args.local_rank is None:
        rank = -1
    else:
        dist_init(args)
        rank = dist_get_rank()

    if rank <= 0:
        # non-distributed or master
        if hasattr(args, 'reset') and args.reset:
            print('Confirm to delete existing checkpoints and logs (Y/N)? ', end='')
            choice = input().upper()
            if choice in {'Y', 'YES'}:
                if isdir(args.exp_dir):
                    print(f'Deleting {args.exp_dir}')
                    rmtree(args.exp_dir)
            else:
                print('The response is N')
        makedirs(args.exp_dir, exist_ok=True)

    log_level = vars(logging)[args.log_level]
    logger = logging.getLogger()

    handlers = [logging.StreamHandler()]
    if rank <= 0:
        # non-distributed or master
        log_path = join(args.exp_dir, f'{task_name}_{get_timestamp()}.log')
        handlers.append(logging.FileHandler(log_path, 'w'))

        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    formatter = LevelSpecificFormatter(rank)
    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.use_cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.use_cuda else 'cpu'
    if not args.use_cuda and args.gpu_gather:
        logging.warning('gpu_gather is turned off since CUDA is off')
        args.gpu_gather = False

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        
    if rank <= 0:
        json.dump(
            vars(args),
            open(join(args.exp_dir, f'{task_name}-args.json'), 'w'),
            sort_keys=True,
            indent=2,
        )

    logging.info(f'Task name: {task_name}')
    if args.log_env_info:
        logging.info(
            'Environment and settings:\n' + \
            get_env_info(use_cuda=args.use_cuda, to_str=True) + \
            f'- exp_dir: {args.exp_dir}'
        )

    return args

def load_ext(module_name):
    ext_paths = environ.get('LLCV_EXT_PATH', None)
    if not ext_paths:
        return None
        
    if os_name == 'nt':
        # Windows
        ext_paths = ext_paths.split(';')
    else:
        ext_paths = ext_paths.split(':')
    # trailing separator could lead to empty items
    ext_paths = list(filter(None, ext_paths))
    if len(ext_paths) == 0:
        return None

    loaded_exts = []
    for path in ext_paths:
        sys.path.insert(0, path)
        try:
            ext = __import__(module_name)
            loaded_exts.append(ext)
            logging.info(f'Loaded extension "{module_name}" from {path}')
        except ModuleNotFoundError:
            pass

    return loaded_exts

def build_ext_class(module_name, class_name, *args, **kargs):
    exts = load_ext(module_name)
    if exts:
        for ext in exts:
            if class_name in vars(ext):
                return vars(ext)[class_name](*args, **kargs)
    return None
