import logging

from llcv.tasks.base_task import BaseTask


class PrintTask(BaseTask):
    def forward(self, data):
        x, y = data
        x = self.model(x)
        logging.info(str(x))
        logging.info(str(y))
        

