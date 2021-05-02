from os import environ

import torch
import torch.distributed as tdist
import torch.multiprocessing as mp

    
def dist_add_args(parser):
    parser.add_argument('--dist-backend', type=str, default='nccl',
        help='distributed backend')
    parser.add_argument('--local_rank', type=int, default=None,
        help='auto filled by torch.distributed.launch')

    '''
    These environment variables will be auto set by torch.distributed.launch:
    MASTER_ADDR, MASTER_PORT, WORLD_SIZE, OMP_NUM_THREADS, RANK, LOCAL_RANK
    '''

def dist_init(args):
    if args.local_rank >= torch.cuda.device_count():
        raise Exception("--nproc_per_node is set larger than the # of visible GPUs")
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    torch.cuda.set_device(args.local_rank)
    tdist.init_process_group(args.dist_backend)

def dist_is_on():
    return tdist.is_available() and tdist.is_initialized()

def dist_get_rank():
    if dist_is_on():
        return tdist.get_rank()
    else:
        return -1

def dist_get_world_size():
    if dist_is_on():
        return tdist.get_world_size()
    else:
        return 1

def dist_get_info():
    if not dist_is_on():
        return {'distributed': False}
        
    return {
        'distributed': True,
        'distributed backend': tdist.get_backend(),
        'world size': tdist.get_world_size(),
        'master addr': environ.get('MASTER_ADDR'),
        'master port': environ.get('MASTER_PORT'),
        'rank': tdist.get_rank(),
        'local rank': environ.get('LOCAL_RANK'),
    }

    