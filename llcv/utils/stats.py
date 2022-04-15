
from time import perf_counter
from datetime import datetime

from dateutil.relativedelta import relativedelta

import numpy as np
import torch


def get_batchsize(data):
    return len(data[0]) if isinstance(data, (list, tuple)) else len(data)

def get_timestamp():
    return str(datetime.now()).replace(' ', '_').replace(':', '-')

def fmt_elapse_time(elapsed):
    # the input unit is second
    attrs = ['days', 'hours', 'minutes', 'seconds']
    delta = relativedelta(seconds=round(elapsed))
    fmtstr = ''
    if delta.days >= 7:
        weeks, delta.days = divmod(delta.days, 7)
        fmtstr += str(weeks) + 'w'
    for a in attrs:
        if getattr(delta, a):
            fmtstr += str(getattr(delta, a)) + a[0]
    if fmtstr == '':
        fmtstr = '0'
    return fmtstr

def get_eta(elapsed, finished, total):
    if elapsed == 0 or finished == 0 or total == finished:
        return '0'
    else:
        return fmt_elapse_time(elapsed*(total - finished)/finished)

def sprint_stats(var, name='', fmt='%.2f', cvt=lambda x: x):
    var = np.asarray(var)
    
    if name:
        prefix = name + ': '
    else:
        prefix = ''

    if len(var) == 1:
        return ('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        )
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        return ('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        )

class Timer():
    def __init__(self):
        self.t_start = perf_counter()

    def stop(self):
        self.t_end = perf_counter()

    def check(self):
        return perf_counter() - self.t_start

    def elapsed(self, to_str=False):
        t_elapsed = self.t_end - self.t_start
        if to_str:
            return fmt_elapse_time(t_elapsed)
        return t_elapsed

    def restart(self):
        self.t_start = perf_counter()

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val*n
        self.count += n
    
    def avg(self):
        return self.sum/self.count

def topk_accu(output, target, topk=(1,), multiplier=100.0):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(multiplier/batch_size).item())
        return res
