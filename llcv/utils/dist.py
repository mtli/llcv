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

def dist_cpu_gather(*variables, tmp_dir=None):
    '''
    Gather variables from all ranks and concatenate them along the batch
    dimension. CPU gather makes use of temporary files. The variables
    can be on either CPU or GPU.
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

    if rank == 0:
        gathered_vars = [[v] for v in variables]
        for i in range(1, world_size):
            vars_part = torch.load(f'rank{i}.pth')
            for v1, v2 in zip(gathered_vars, vars_part):
                v1.append(v2)
        # the tmp dir is no longer needed
        rmtree(tmp_dir)
        for i, vars_all in enumerate(gathered_vars):
            gathered_vars[i] = torch.cat(vars_all)
        return gathered_vars
    else:
        return None

def dist_gpu_gather(*variables):
    '''
    Gather variables from all ranks and concatenate them along the batch
    dimension. GPU gather makes use of collective operations. All
    variables must be on the GPU. Note that due to potential uneven data
    slicing among distributed processes, we need to first communicate to
    get the number of examples held by each process before allocating
    the receiving buffers.
    '''

    world_size = tdist.get_world_size()

    n_exp = len(variables[0])
    device = variables[0].device

    # synchronize about the number of examples
    n_exp = torch.tensor(n_exp, dtype=torch.long, device=device)
    n_exp_list = [torch.empty_like(n_exp) for _ in range(world_size)]
    tdist.all_gather(n_exp_list, n_exp)
    n_exp_max = torch.tensor(n_exp_list).max()

    # pad the variables according to the max number of examples so that they
    # are of the same shape on all processes
    vars_padded = []
    for v in variables:
        v_padded = torch.empty((n_exp_max, *v.shape[1:]), dtype=v.dtype, device=device)
        v_padded[:n_exp] = v
        vars_padded.append(v_padded)

    # TODO：seralize (packing) all variables before communication

    # In theory, gathering needs only to be done on rank 0. But NCCL only
    # supports all_gather(), not gather(). Therefore we gather on all ranks here.
    gathered_vars = []
    for v_padded in vars_padded:
        # allocate the buffers
        buffers = [torch.empty_like(v_padded) for _ in range(world_size)]
        # communicate
        tdist.all_gather(buffers, v_padded)
        # remove the paddings
        vars_all = [v_padded[:n_exp] for n_exp, v_padded in zip(n_exp_list, buffers)]
        gathered_vars.append(torch.cat(vars_all))

    return gathered_vars
    