import logging, warnings, json
from abc import ABCMeta, abstractmethod
from os import rename, makedirs
from os.path import join, basename, isfile
from glob import glob

import numpy as np

import torch
import torch.optim as optims
import torch.optim.lr_scheduler as lr_schedulers
from torch.nn import DataParallel as DP
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from linearlr import LinearLR

from ..models import build_model
from ..utils import dist_get_rank, sprint_stats

class BaseTask(metaclass=ABCMeta):
    def __init__(self, args, loader, is_train):
        self.loader = loader
        self.dataset = loader.dataset
        self.is_train = is_train
        self.device = args.device
        self.gather = False
        self.gpu_gather = args.gpu_gather
        self.resume_epoch = 0
        self.has_val_score = False
        self.exp_dir = args.exp_dir
        if self.is_train:
            self.last_lr = args.lr
            self.lr_update_per_epoch = args.lr_update_per_epoch

        self.model = build_model(args, self.dataset)
        logging.debug(str(self.model))
        logging.debug(f'Total number of parameters: {sum([p.numel() for p in self.model.parameters()])}')

        self.rank = dist_get_rank()
        if self.rank >= 0:
            self.device = torch.cuda.current_device() if args.use_cuda else 'cpu'
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                [self.device] if args.use_cuda else None,
                find_unused_parameters=True,
            )
        else:
            if args.use_cuda:
                if torch.cuda.device_count() > 1:
                    self.model = DP(self.model)
                self.model = self.model.to(self.device)
        self.output_device = args.device if args.gpu_gather else 'cpu'
    
        if is_train:
            logging.debug(f'Optimizer: {args.optim} with base learning rate {args.lr:.6g}')
            self.set_optim(args)
            self.set_lr_schedule(args)

        self.auto_load(args)

    def set_optim(self, args):
        if args.optim == 'SGD':
            self.optim = optims.SGD(self.model.parameters(),
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        elif args.optim == 'Adam':
            self.optim = optims.Adam(self.model.parameters(),
                lr=args.lr, weight_decay=args.wd)
        else:
            self.optim = vars(optims)[args.optim](self.model.parameters(), lr=args.lr)
        
    def set_lr_schedule(self, args):
        if args.lr_schedule:
            if args.lr_update_per_epoch:
                if args.lr_schedule == 'Linear':
                    self.lr_scheduler = LinearLR(
                        self.optim,
                        T=args.n_epoch,
                        last_epoch=self.resume_epoch - 1,
                    )
                elif args.lr_schedule == 'CosineAnnealing':
                    self.lr_scheduler = lr_schedulers.CosineAnnealingLR(
                        self.optim,
                        T_max=args.n_epoch,
                        last_epoch=self.resume_epoch - 1,
                    )
                elif args.lr_schedule == 'Step':
                    self.lr_scheduler = lr_schedulers.StepLR(self.optim,
                        step_size=args.lr_decay_step_size,
                        gamma=args.lr_schedule_gamma,
                        last_epoch=self.resume_epoch - 1,
                    )
                elif args.lr_schedule == 'MultiStep-ImageNet':
                    self.lr_scheduler = lr_schedulers.MultiStepLR(self.optim,
                        milestones=[round(0.3*args.n_epoch), round(0.6*args.n_epoch)],
                        gamma=args.lr_schedule_gamma,
                        last_epoch=self.resume_epoch - 1,
                    )
                elif args.lr_schedule == 'MultiStep-COCO':
                    self.lr_scheduler = lr_schedulers.MultiStepLR(self.optim,
                        milestones=[round(0.6*args.n_epoch), round(0.9*args.n_epoch)],
                        gamma=args.lr_schedule_gamma,
                        last_epoch=self.resume_epoch - 1,
                    )
                else:
                    self.lr_scheduler = vars(lr_schedulers) \
                        [args.lr_schedule + 'LR'](self.optim,
                            last_epoch=self.resume_epoch - 1,
                        )
            else:
                # using the last_epoch as a counter for number of iters processed so far
                n_iter_epoch = len(self.loader)
                n_iter_total = args.n_epoch*n_iter_epoch
                last_counter = self.resume_epoch*n_iter_epoch - 1
                if args.lr_schedule == 'Linear':
                    self.lr_scheduler = LinearLR(self.optim,
                        T=n_iter_total,
                        last_epoch=last_counter,
                    )
                elif args.lr_schedule == 'CosineAnnealing':
                    self.lr_scheduler = lr_schedulers.CosineAnnealingLR(
                        self.optim,
                        T_max=n_iter_total,
                        last_epoch=last_counter,
                    )
                elif args.lr_schedule == 'Step':
                    self.lr_scheduler = lr_schedulers.StepLR(self.optim,
                        step_size=args.lr_decay_step_size,
                        gamma=args.lr_schedule_gamma,
                        last_epoch=last_counter,
                    )
                else:
                    self.lr_scheduler = vars(lr_schedulers) \
                        [args.lr_schedule + 'LR'](self.optim,
                            last_epoch=last_counter,
                        )                

    def get_lr(self):
        if not hasattr(self, 'lr_scheduler'):
            return self.lr_scheduler.get_last_lr()[0]
        else:
            return self.optim.param_groups[0]['lr']

    def update_lr_epoch(self):
        if not hasattr(self, 'lr_scheduler'):
            return
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        if self.last_lr != lr:
            logging.info(f'Base learning rate updated to {lr:.6g}')
            self.last_lr = lr

    def update_lr_iter(self):
        if not hasattr(self, 'lr_scheduler'):
            return
        self.lr_scheduler.step()
            
    def train_mode(self, gather=True):
        self.model.train()
        self.gather = gather

    def test_mode(self, gather=True):
        self.model.eval()
        self.gather = gather

    def mark_best_model(self, epoch, score):
        with open(join(self.exp_dir, 'best-model.txt'), 'w') as f:
            f.write('%d\n%g\n' % (epoch, score))

    def query_best_model(self):
        file_path = join(self.exp_dir, 'best-model.txt')
        if not isfile(file_path):
            return None, None
        with open(file_path) as f:
            content = f.readlines()
        epoch = int(content[0].strip())
        score = float(content[1].strip())
        return epoch, score

    def save(self, epoch):
        if self.rank > 0:
            return
        out_dict = {}
        if isinstance(self.model, (DP, DDP)):
            out_dict['model'] = self.model.module.state_dict()
        else:
            out_dict['model'] = self.model.state_dict()
        out_dict['optim'] = self.optim.state_dict()
        if hasattr(self, 'lr_scheduler'):
            out_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        # using a temporary file first to prevent getting
        # corrupted saves in case of a system crash
        tmp_path = join(self.exp_dir, 'temp.pth')
        torch.save(out_dict, tmp_path)
        rename(tmp_path, join(self.exp_dir, 'e%03d.pth' % epoch))

    def load(self, path, strict=False):
        if isinstance(self.device, int):
            device = f'cuda:{self.device}'
        else:
            device = self.device
        ckpt = torch.load(path, map_location=device)
        if 'model' in ckpt:
            # llcv format
            model_state_dict = ckpt['model']
            if hasattr(self, 'optim') and 'optim' in ckpt:
                self.optim.load_state_dict(ckpt['optim'])
            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in ckpt:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        else:
            # simple model-state-dict format
            model_state_dict = ckpt
        
        # convert both the model and the state_dict to the basic format without DP or DDP wrappers
        if next(iter(model_state_dict)).startswith('module.'):
            model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
        if isinstance(self.model, (DP, DDP)):
            model = self.model.module
        else:
            model = self.model
        model.load_state_dict(model_state_dict, strict)

    def auto_load(self, args):
        '''
        Automatically finding out which model to load given the arguments
        '''
        self.resume_epoch = 0
        strict = args.load_strict
        if self.is_train:
            if args.resume_epoch is not None:
                self.resume_epoch = args.resume_epoch
                if args.resume_load:
                    ckpt_path = join(self.exp_dir, 'e%03d.pth' % self.resume_epoch)
                    logging.warning('Resume training from ' + ckpt_path)
                    self.load(ckpt_path, strict)
                elif args.pretrain:
                    logging.info('Loading pretrained model ' + args.pretrain)
                    self.load(args.pretrain, strict)
            else:
                saves = glob(join(args.exp_dir, 'e*.pth'))
                if saves:
                    # previously trained, loading the latest model
                    self.resume_epoch = max([int(basename(s).split('.')[0][1:]) for s in saves])
                    ckpt_path = join(self.exp_dir, 'e%03d.pth' % self.resume_epoch)
                    logging.warning('Resume training from ' + ckpt_path)
                    self.load(ckpt_path, strict)
                elif args.pretrain:
                    logging.info('Loading pretrained model ' + args.pretrain)
                    self.load(args.pretrain, strict)
        else:
            if args.test_init:
                return
            if args.ckpt:
                ckpt_path = args.ckpt
                logging.info('Loading checkpoint ' + ckpt_path)
            elif args.ckpt_epoch:
                self.resume_epoch = args.ckpt_epoch
                ckpt_path = join(self.exp_dir, 'e%03d.pth' % self.resume_epoch)
                logging.info('Loading checkpoint with specified epoch ' + ckpt_path)
            else:
                self.resume_epoch, score = self.query_best_model()
                if self.resume_epoch is not None:
                    ckpt_path = join(self.exp_dir, 'e%03d.pth' % self.resume_epoch)
                    logging.info(f'Loading the best checkpoint (score {score:.6g}) from {ckpt_path}')
                else:
                    saves = glob(join(args.exp_dir, 'e*.pth'))
                    if not saves:
                        raise Exception('Cannot find saved models in ' + args.exp_dir)
                    self.resume_epoch = max([int(basename(s).split('.')[0][1:]) for s in saves])
                    ckpt_path = join(self.exp_dir, 'e%03d.pth' % self.resume_epoch)
                    logging.info(f'Loading the latest model from {ckpt_path}')
            self.load(ckpt_path, strict)

        if self.resume_epoch is None:
            self.resume_epoch = 0

    def summarize_timing(self, timing_type, samples, n_warmup, out_dir):
        assert len(samples) > n_warmup, 'Not enough timing samples after warming up'
        samples = 1e3*np.asarray(samples)
        logging.info(sprint_stats(samples[n_warmup:], timing_type.capitalize() + ' (ms)'))
        if out_dir:
            makedirs(out_dir, exist_ok=True)
            out_dict = {
                timing_type: {
                    'unit': 'ms',
                    'n_warmup': n_warmup,
                    'mean': samples[n_warmup:].mean(),
                    'std': samples[n_warmup:].std(ddof=1),
                    'min': samples[n_warmup:].min(),
                    'max': samples[n_warmup:].max(),
                    'samples': samples.tolist(),
                },
            }
            out_path = join(out_dir, timing_type.replace(' ', '_') + '.json')
            logging.info(f'Saving timing information to {out_path}')
            json.dump(out_dict, open(out_path, 'w'), indent=2)

    @abstractmethod
    def forward(self, data):
        pass

    def backward(self, data):
        pass

    def log_iter(self, str_prefix='', str_suffix=''):
        pass

    def log_iter_tb(self, accu_iter, is_train):
        pass
        
    def log_epoch(self, str_prefix='', str_suffix=''):
        pass

    def log_epoch_tb(self, epoch, is_train):
        pass

    def dist_gather(self, is_train):
        pass

    def get_test_scores(self, force_update=False):
        pass

    def summarize_test(self, args):
        pass
        