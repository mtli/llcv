import logging

import torch

from ..datasets import build_loader
from ..tasks import build_task
from ..utils import get_default_parser, env_setup, \
    Timer, get_eta, dist_get_world_size, get_batchsize


def add_args(parser):
    ## Basic options
    parser.add_argument('--dataset', type=str, default='CIFAR10',
        help='dataset')
    parser.add_argument('--data-root', type=str, required=True,
        help='root directory of the dataset')
    parser.add_argument('--n-epoch', type=int, default=20,
        help='# of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size for training (per node)')
    parser.add_argument('--n-worker', type=int, default=8,
        help='# of workers for data prefetching (per node)')
    parser.add_argument('--lr', type=float, default=0.1,
        help='base learning rate (default: 0.1)')

    ## Hyperparameters
    parser.add_argument('--optim', type=str, default='SGD',
        help='optimizer (default: SGD)')
    parser.add_argument('--wd', type=float, default=5e-4,
        help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='optimizer momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', default=False,
        help='enables nesterov momentum')
    parser.add_argument('--lr-schedule', type=str, default='Linear',
        help='learning rate schedule (default: Linear)')
    parser.add_argument('--lr-update-per-epoch', action='store_true', default=False,
        help='update learning rate after each epoch instead of each iter by default')
    parser.add_argument('--lr-decay-epoch', type=int, default=50,
        help='learning rate schedule (default: 50)')
    parser.add_argument('--lr-schedule-gamma', type=float, default=0.1,
        help='intepretation depends on lr_schedule (default: 0.1)')

    ## Training Settings
    parser.add_argument('--reset', action='store_true', default=False,
        help='DANGER: purge the exp_dir and start a fresh new training run')
    parser.add_argument('--pretrain', type=str, default=None,
        help='pretrained weights')
    parser.add_argument('--resume-epoch', type=int, default=None,
        help='by default, the resume epoch is automatically determined from '
             'the checkpoint file and this option overwrites the default')
    parser.add_argument('--no-resume-load', action='store_false', dest='resume_load', default=True,
        help='resume from an epoch, but not load the checkpoint')
    parser.add_argument('--batch-size-per-gpu', type=int, default=None,
        help='alternative to batch_size (and overrides it)')
    parser.add_argument('--n-worker-per-gpu', type=int, default=None,
        help='alternative n_worker (and overrides it)')
    parser.add_argument('--epoch-iter', type=int, default=float('inf'),
        help='maximum # of iterations per epoch')
    parser.add_argument('--no-val', action='store_false', dest='val', default=True,
        help='turn off validation')
    parser.add_argument('--log-interval', type=int, default=50,
        help='after every how many iters to log the training status')
    parser.add_argument('--save-interval', type=int, default=5,
        help='after every how many epochs to save the learned model')
    parser.add_argument('--val-interval', type=int, default=5,
        help='after every how many epochs to save the learned model')
    parser.add_argument('--train-gather', action='store_true', default=False,
        help='gather results over batches during training, which is required '
             'to compute metrics over the entire training set at the end of '
             'every epoch',
    )

def main():
    ## Overall timer
    tmr_main = Timer()

    ## Argument parser and environment setup
    parser = get_default_parser('llcv - training script')
    add_args(parser)
    args = env_setup(parser, 'train', ['data_root', 'pretrain'])

    ## Prepare the dataloader
    train_loader = build_loader(args, is_train=True)
    logging.info(f'# of classes: {len(train_loader.dataset.classes)}')
    n_train = len(train_loader.dataset)
    logging.info(f'# of training examples: {n_train}')
    assert n_train
    if args.epoch_iter < len(train_loader):
        logging.warning(
            f'The number of iterations per epoch is limited to {args.epoch_iter}')
        train_epoch_iter = args.epoch_iter
    else:
        train_epoch_iter = len(train_loader)

    if args.val:
        val_loader = build_loader(args, is_train=False)
        n_val = len(val_loader.dataset)
        logging.info(f'# of validation examples: {n_val}')
    else:
        n_val = 0

    ## Initialize task
    task = build_task(args, train_loader, is_train=True)

    if task.resume_epoch >= args.n_epoch:
        logging.warning(f'The model is already trained for {task.resume_epoch} epochs')
        return

    if n_val and task.has_val_score:
        if task.resume_epoch:
            best_epoch, best_score = task.query_best_model()
        else:
            best_score = best_epoch = 0

    ## Start training
    last_saved_epoch = task.resume_epoch # resume_epoch is by default 0
    # counters for ETA
    n_iter_epoch = 0
    n_iter_total = (args.n_epoch - task.resume_epoch)*train_epoch_iter
    # scaling factor for DDP
    speed_ratio = dist_get_world_size()
    
    logging.info('Training starts')
    tmr_train = Timer()
    for epoch in range(task.resume_epoch + 1, args.n_epoch + 1):
        task.train_mode(args.train_gather)

        n_seen = 0
        n_warpup = 0
        t_warmup = 0
        tmr_epoch = Timer()
        for i, data in enumerate(train_loader):
            i += 1
            # the last batch can be smaller than normal
            this_batch_size = get_batchsize(data)

            tmr_iter = Timer()
            task.forward(data)
            task.backward()
            tmr_iter.stop()

            if not args.lr_update_per_epoch:
                task.update_lr_iter()

            n_seen += this_batch_size

            t_iter = tmr_iter.elapsed()
            if i <= args.timing_warmup_iter:
                n_warpup += this_batch_size
                t_warmup += t_iter

            if i % args.log_interval == 0:
                t_total = tmr_epoch.check()
                if i <= args.timing_warmup_iter:
                    ave_speed = n_seen/t_total if t_total else float('inf')
                else:
                    ave_speed = (n_seen - n_warpup)/(t_total - t_warmup)if (t_total - t_warmup) else float('inf')
                ave_speed *= speed_ratio

                task.log_iter(
                    'train e%d: %4d/%4d, %5.4gHz, ' % 
                    (epoch, i, train_epoch_iter, ave_speed),
                    ', ETA: ' + get_eta(tmr_train.check(), n_iter_epoch + i, n_iter_total)
                )
                  
                task.log_iter_tb(
                    (epoch-1)*train_epoch_iter + i,
                    is_train=True,
                )

            if i >= train_epoch_iter:
                break

        task.dist_gather(is_train=True)
        task.log_epoch(f'train e{epoch} summary: ')
        task.log_epoch_tb(epoch, is_train=True)
        task.reset_epoch()      

        if n_val and (epoch % args.val_interval == 0 or epoch == args.n_epoch):
            n_seen = 0
            task.test_mode()

            n_warpup = 0
            t_warmup = 0
            tmr_val = Timer()
            for i, data in enumerate(val_loader):
                i += 1
                this_batch_size = get_batchsize(data)
                
                tmr_iter = Timer()
                with torch.no_grad():
                    task.forward(data)
                tmr_iter.stop()

                n_seen += this_batch_size

                t_iter = tmr_iter.elapsed()
                if i <= args.timing_warmup_iter:
                    n_warpup += this_batch_size
                    t_warmup += t_iter

                if i % args.log_interval == 0:
                    t_total = tmr_val.check()
                    if i <= args.timing_warmup_iter:
                        ave_speed = n_seen/t_total if t_total else float('inf')
                    else:
                        ave_speed = (n_seen - n_warpup)/(t_total - t_warmup)if (t_total - t_warmup) else float('inf')
                    ave_speed *= speed_ratio

                    task.log_iter(
                        'val e%d: %4d/%4d, %6.5gHz, ' % 
                        (epoch, i, len(val_loader), ave_speed),
                    )

            task.dist_gather(is_train=False)
            if task.has_val_score:
                new_score = task.get_test_scores()[0]
                if new_score > best_score:
                    best_score = new_score
                    best_epoch = epoch
                    task.mark_best_model(best_epoch, best_score)
                    task.save(epoch)
                    last_saved_epoch = epoch
            task.log_epoch(f'val e{epoch} summary: ')
            task.log_epoch_tb(epoch, is_train=False)
            task.reset_epoch()

        tmr_epoch.stop()
        logging.info('end of epoch %d/%d: epoch time: %s, epoch-based ETA: %s' %
                (epoch, args.n_epoch, tmr_epoch.elapsed(to_str=True),
                get_eta(tmr_train.check(),
                    epoch - task.resume_epoch,
                    args.n_epoch - task.resume_epoch,
                ))
        )
        
        if last_saved_epoch != epoch and epoch % args.save_interval == 0:
            task.save(epoch)
            last_saved_epoch = epoch
            
        if args.lr_update_per_epoch:
            task.update_lr_epoch()

        n_iter_epoch += train_epoch_iter

    if last_saved_epoch != args.n_epoch:
        # saving the last epoch if n_epoch is not divisible by save_interval
        task.save(args.n_epoch)
    
    tmr_main.stop()
    logging.info(f'Training finished with total elapsed time {tmr_main.elapsed(to_str=True)}')

    if n_val and task.has_val_score:
        logging.info(f'The best model is obtained at epoch {best_epoch} with score {best_score:.6g}')

if __name__ == '__main__':
    main()
