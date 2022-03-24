from os import environ, makedirs
from tempfile import mkdtemp
from shutil import rmtree

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

def _reorder_sliced_data(buffers, dataset_size):
    '''
    Undo the slicing done by torch.utils.data.distributed.DistributedSampler
    For example, for distributed world of size 4, these are the data
    processed by each rank:
    rank 0: [0, 4, 8, ...]
    rank 1: [1, 5, 9, ...]
    rank 2: [2, 6, 10, ...]
    rank 3: [3, 7, 11, ...]

    This function assumes that the data loader sampler uses
    DistributedSampler or the same logic as in DistributedSampler to slice
    the data. Also it removes padded examples by DistributedSampler
    according to the input dataset_size.

    buffers: [buf_rank0, buf_rank1, ...]

    '''
    var_ordered = []
    for var_parts in zip(*buffers):
        var_ordered.extend(list(var_parts))
    return var_ordered[:dataset_size]

def dist_cpu_gather(variables, dataset_size, tmp_dir=None):
    '''
    Gather variables from all ranks and concatenate them along the batch
    dimension. CPU gather makes use of temporary files. The variables
    can be on either the CPU or the GPU. The output variables are on the
    CPU. This function assumes
    torch.utils.data.distributed.DistributedSampler is used in the data
    loader for data sampling. Specifically, the examples indices
    should be duplicated to ensure the same number of examples across
    all ranks in an epoch, and the data slicing should follow this
    logic: [rank: dataset_size: world_size].
    '''

    rank = dist_get_rank()
    world_size = tdist.get_world_size()
    
    if tmp_dir is None:
        # we first create a directory to save all
        # the temporary files (as a hidden directory)
        tmp_parent = '.dist_cpu_gather'

        # The need of a two-level hierarchy is because multiple llcv jobs
        # may run at the same time and each job requires a unique folder.
        # Here we create the folder on rank 0 and broadcast to other
        # ranks.
        MAX_FILENAME = 255

        # create a buffer to receive the unqiue folder name
        dir_name_buf = torch.full(
            (len(tmp_parent) + 1 + MAX_FILENAME,),
            32, # The ASCII/UTF-8 code for space
            dtype=torch.uint8,
            device='cuda',
        )

        if rank == 0:
            makedirs(tmp_parent, exist_ok=True)
            tmp_dir = mkdtemp(dir=tmp_parent)

            # by default .encode() returns the immutable type of bytes,
            # and it needs to be converted to bytearray to work with 
            # PyTorch
            tmp_dir_b = bytearray(tmp_dir.encode())
            tmp_dir_b = torch.tensor(tmp_dir_b,
                dtype=torch.uint8, device='cuda')

            dir_name_buf[:len(tmp_dir_b)] = tmp_dir_b
        # broadcast from rank 0 to other ranks
        tdist.broadcast(dir_name_buf, 0)
        # recover the dir path
        tmp_dir = dir_name_buf.cpu().numpy().tobytes().decode().rstrip()

    else:
        # a working directory for dist_cpu_gather is given
        makedirs(tmp_dir, exist_ok=True)

    torch.save(variables, f'rank{rank}.pth')
    # wait for all ranks to finish saving
    tdist.barrier()

    if rank != 0:
        return None

    buffers = [[v.cpu()] + (world_size-1)*[None] for v in variables]
    for i in range(1, world_size):
        vars_part = torch.load(f'rank{i}.pth', map_location='cpu')
        for v_buf, var_part in zip(buffers, vars_part):
            v_buf[i] = var_part

    # the tmp dir is no longer needed
    rmtree(tmp_dir)

    # restore the original dataset order
    gathered_vars = []
    for v_buf in buffers:
        # restore the original dataset order
        v_ordered = _reorder_sliced_data(v_buf, dataset_size)
        # bring back the batch dimension and concatenate
        gathered_vars.append(torch.cat([x.unsqueeze(0) for x in v_ordered]))

    return gathered_vars

def dist_gpu_gather(variables, dataset_size):
    '''
    Gather variables from all ranks and concatenate them along the batch
    dimension. GPU gather makes use of collective operations. All
    variables must be on the GPU. This function makes the same
    assumption about data sampling as in dist_cpu_gather().
    '''
    # TODO：seralize (packing) all variables before communication

    # In theory, gathering needs only to be done on rank 0. But NCCL only
    # supports all_gather(), not gather(). Therefore we gather on all ranks here.
    world_size = tdist.get_world_size()
    gathered_vars = []
    for v in variables:
        # allocate the buffers
        v_buf = [torch.empty_like(v) for _ in range(world_size)]
        # communicate
        tdist.all_gather(v_buf, v)
        # restore the original dataset order
        v_ordered = _reorder_sliced_data(v_buf, dataset_size)
        # bring back the batch dimension and concatenate
        gathered_vars.append(torch.cat([x.unsqueeze(0) for x in v_ordered]))

    return gathered_vars
    