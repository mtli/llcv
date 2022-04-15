import logging, json
from os import makedirs
from os.path import join

import torch
import torch.nn as nn
import torch.distributed as tdist
from torch.utils.tensorboard import SummaryWriter

# from torchvision.utils import make_grid

import numpy as np
# from PIL import Image
# from html4vision import Col, imagetable

from .base_task import BaseTask
from ..utils import AverageMeter, topk_accu, dist_cpu_gather, dist_gpu_gather


class ClsTask(BaseTask):
    def __init__(self, args, loader, is_train):
        super().__init__(args, loader, is_train)
        self.test_no_gt = hasattr(self.dataset, 'no_gt') and self.dataset.no_gt
        self.num_classes = len(self.dataset.classes)
        self.has_val_score = not self.test_no_gt

        if not self.test_no_gt:
            if not hasattr(args, 'loss') or args.loss is None:
                self.loss_name = 'CrossEntropy'
                self.criterion = nn.CrossEntropyLoss().to(self.device)
            else:
                self.loss_name = args.loss
                self.criterion = vars(nn)[self.loss_name + 'Loss']().to(self.device)
            logging.debug(f'Criterion: {self.loss_name}')

        if is_train:
            if self.rank <= 0:
                self.writer = SummaryWriter(join(args.exp_dir, 'tb'))
        self.reset_epoch()

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()
            
    def reset_epoch(self):
        # epoch-wise stats
        self.avg_loss = AverageMeter()
        self.accu_all = None
        
        # outputs to be gathered
        self.y_all = torch.empty((0,), dtype=torch.long, device=self.output_device)
        self.y_out_all = torch.empty((0, self.num_classes), dtype=torch.float, device=self.output_device)

    def forward(self, data):
        x, y = data
        if not isinstance(self.model, nn.DataParallel):
            # DataParallel's broadcast is much faster than
            # manually moving data to the first GPU
            x = x.to(self.device)
        if y.is_cuda:
            if self.gather and not self.gpu_gather:
                y_cpu = y.cpu()            
        else:
            y_cpu = y
            y = y.to(self.device)

        y_out = self.model(x)
        self.loss = self.criterion(y_out, y)
        self.avg_loss.update(self.loss.item(), len(x))

        self.accu = topk_accu(y_out, y, (1, 5))
        
        if self.gather:
            if self.gpu_gather:
                self.y_all = torch.cat([self.y_all, y])
                self.y_out_all = torch.cat([self.y_out_all, y_out])
            else:
                self.y_all = torch.cat([self.y_all, y_cpu])
                self.y_out_all = torch.cat([self.y_out_all, y_out.cpu()])

    def backward(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    def log_iter(self, str_prefix='', str_suffix=''):
        logging.info(f'{str_prefix}loss: {self.loss.item():5.3g}, top-1: {self.accu[0]:5.4g}%, top-5: {self.accu[1]:5.4g}%{str_suffix}')

    def log_iter_tb(self, total_iter, is_train):
        if self.rank > 0:
            return
        name_suffix = '_Train' if is_train else '_Val'
        self.writer.add_scalar('Iter/' + self.loss_name + name_suffix, self.loss.item(), total_iter)
        self.writer.add_scalar('Iter/Top-1' + name_suffix, self.accu[0], total_iter)
        self.writer.add_scalar('Iter/Top-5' + name_suffix, self.accu[1], total_iter)
        if is_train and not self.lr_update_per_epoch:
            self.writer.add_scalar('Iter/LR', self.get_lr(), total_iter)

    def log_epoch(self, str_prefix='', str_suffix=''):
        if self.gather:
            accu = self.get_test_scores()
            logging.info(f'{str_prefix}loss: {self.avg_loss.avg():5.3g}, top-1: {accu[0]:5.4g}%, top-5: {accu[1]:5.4g}%{str_suffix}')
        else:
            logging.info(str_prefix + str_suffix)

    def log_epoch_tb(self, epoch, is_train):
        if self.rank > 0 or not self.gather:
            return
        name_suffix = '_Train' if is_train else '_Val'
        self.writer.add_scalar('Epoch/' + self.loss_name + name_suffix, self.avg_loss.avg(), epoch)
        accu = self.get_test_scores()
        self.writer.add_scalar('Epoch/Top-1' + name_suffix, accu[0], epoch)
        self.writer.add_scalar('Epoch/Top-5' + name_suffix, accu[1], epoch)
        if is_train and self.lr_update_per_epoch:
            self.writer.add_scalar('Epoch/LR', self.get_lr(), epoch)

    def dist_gather(self, is_train):
        '''
        Gather outputs from all processes in the distributed setting
        '''
        if self.rank < 0 or not self.gather:
            return

        dist_func = dist_gpu_gather if self.gpu_gather else dist_cpu_gather
        gathered = dist_func((self.y_all, self.y_out_all), len(self.dataset))

        if self.rank == 0:
            self.y_all, self.y_out_all = gathered
        
    def get_test_scores(self, force_update=False):
        # the primary metric should be placed in the first place
        if force_update or self.accu_all is None:
            self.accu_all = topk_accu(self.y_out_all, self.y_all, (1, 5))
        return self.accu_all

    def summarize_test(self, args):
        if self.rank > 0:
            return
        if not self.gather:
            logging.warning('Gather is disabled and therefore the results are not summarized or saved')
            return

        loss = self.avg_loss.avg()
        accu = self.get_test_scores()
        logging.info(f'Test summary: loss: {loss:.6g}, top-1: {accu[0]:.6g}%, top-5: {accu[1]:.6g}%')
        
        out_dir = args.out_dir
        if out_dir:
            makedirs(out_dir, exist_ok=True)

            out_dict = {
                'loss': loss,
                'top-1': accu[0],
                'top-5': accu[1],
            }
            out_path = join(out_dir, 'eval.json')
            logging.info(f'Saving evaluation to {out_path}')
            json.dump(out_dict, open(out_path, 'w'))

            pred_all = self.y_out_all.max(1)[1].cpu()
            out_path = join(out_dir, 'pred.txt')
            logging.info(f'Saving predictions to {out_path}')
            np.savetxt(out_path, pred_all.numpy(), '%d')
