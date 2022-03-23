import logging, json, pickle
from os import makedirs
from os.path import join
from copy import deepcopy

import numpy as np

from pycocotools.cocoeval import COCOeval

from torch import nn

from .base_task import BaseTask


class DetTask(BaseTask):
    def __init__(self, args, loader, is_train):
        super().__init__(args, loader, is_train)
        assert not is_train, 'Training is not implemented'
        assert self.rank == -1, 'Distributed is not supported'
        self.test_no_gt = hasattr(self.dataset, 'no_gt') and self.dataset.no_gt
        self.num_classes = len(self.dataset.classes)
        self.has_val_score = not self.test_no_gt

        self.reset_epoch()
            
    def reset_epoch(self):
        # epoch-wise stats
        self.eval_summary = None
        
        self.results_ccf = []

    def forward(self, data):
        x_cpu, y_cpu = data
        if isinstance(self.model, nn.DataParallel):
            # DataParallel's broadcast is much faster than
            # manually moving data to the first GPU
            x = x_cpu
        else:
            x = x_cpu.to(self.device)

        y_out = self.model(x)

        if self.gather:
            for iid, pred in zip(y_cpu['image_id'], y_out):
                iid = iid.item()

                bboxes = pred['boxes'].cpu()
                if len(bboxes) == 0:
                    continue
                # ltrb2ltwh
                bboxes[:, 2:] -= bboxes[:, :2]
                bboxes = bboxes.tolist()
                scores = pred['scores'].cpu().tolist()
                labels = pred['labels'].cpu().tolist()

                for i in range(len(bboxes)):
                    self.results_ccf.append({
                        'image_id': iid,
                        'bbox': bboxes[i],
                        'score': scores[i],
                        'category_id': labels[i],
                    })

    def log_iter(self, str_prefix='', str_suffix=''):
        logging.info(f'{str_prefix}{str_suffix}')

    def eval_ccf(self, results_ccf):
        '''
        Evaluate COCO-format detections
        '''
        db = self.dataset.db
        # COCO utility will also modify the result list
        results_ccf = deepcopy(results_ccf)
        results = db.loadRes(results_ccf)
        cocoEval = COCOeval(db, results, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return {'eval': cocoEval.eval, 'stats': cocoEval.stats}

    def get_test_scores(self, force_update=False):
        # the primary metric should be placed in the first place
        if force_update or self.eval_summary is None:
            self.eval_summary = self.eval_ccf(self.results_ccf)
        return 100*self.eval_summary['stats'][0]

    def summarize_test(self, args):
        if self.rank > 0:
            return
        if not self.gather:
            logging.warning('Gather is disabled and therefore the results are not summarized or saved')
            return
        
        self.get_test_scores()

        out_dir = args.out_dir
        if out_dir:
            makedirs(out_dir, exist_ok=True)

            out_path = join(out_dir, 'AP.txt')
            logging.info(f'Saving APs to {out_path}')
            np.savetxt(out_path, 100*self.eval_summary['stats'][:6], '%.6g')

            out_path = join(out_dir, 'eval.pkl')
            logging.info(f'Saving evaluation to {out_path}')
            pickle.dump(self.eval_summary, open(out_path, 'wb'))

            out_path = join(out_dir, 'results_ccf.json')
            logging.info(f'Saving predictions to {out_path}')
            json.dump(self.results_ccf, open(out_path, 'w'))
