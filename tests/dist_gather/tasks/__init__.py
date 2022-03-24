import logging

from llcv.tasks.base_task import BaseTask
from llcv.utils.dist import dist_cpu_gather, dist_gpu_gather

import torch


class PrintTask(BaseTask):
    def forward(self, data):
        x = self.model(data)
        print(self.rank, x.cpu().numpy().tolist())
        if not hasattr(self, 'x_all'):
            self.x_all = torch.empty((0,), dtype=torch.long, device='cuda')
        self.x_all = torch.cat((self.x_all, x))
        
    def dist_gather(self, is_train):
        '''
        Gather outputs from all processes in the distributed setting
        '''
        if self.rank < 0 or not self.gather:
            return

        dist_func = dist_gpu_gather if self.gpu_gather else dist_cpu_gather
        gathered = dist_func((self.x_all,), len(self.dataset))

        if self.rank == 0:
            self.x_all = gathered[0]

    def summarize_test(self, args):
        if self.rank > 0:
            return
        if not self.gather:
            logging.warning('Gather is disabled and therefore the results are not summarized or saved')
            return

        print(self.x_all.cpu().numpy().tolist())
